import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
import datasets
import neptune
from loss_new import SGTLoss

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from metrics import cos_metric, topk_intersection_metric, reconstruction_rank_metric, evaluate_utility, nn_fr_metric

from collections import defaultdict

from dataset import SGTDataset

from sgt_model import SGTModel


def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return (total_norm ** 0.5)


def compute_gradient_norm_per_layer(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms[f"grad_norm/{name}"] = grad_norm
    
    return grad_norms


def embed_and_split_batch(llm, batch, device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    with torch.no_grad():
        embeds = llm.get_input_embeddings()(input_ids)
    
    B = embeds.size(0)
    if B % 2 != 0:
        embeds = embeds[:-1]
        attention_mask = attention_mask[:-1]
        B = B - 1
    
    if B < 2:
        raise Exception("Can't proceed")
        
    B_half = B // 2
    x = embeds[:B_half]
    x_independent = embeds[B_half:B_half*2]
    mask = attention_mask[:B_half]

    return input_ids[:B_half], x, x_independent, mask


def compute_separate_metrics(llm, tokenizer, sgt, device, mcq_datasets, utility_baseline):
    utility = evaluate_utility(llm, tokenizer, mcq_datasets, device, sgt=sgt)

    return {
        "utility": utility - utility_baseline
    }

def compute_batch_metrics(llm, tokenizer, sgt, input_ids, clean_embeddings, mu, logvar, obfuscated_embeddings, attention_mask, clean_logits, obfuscated_logits, device, mcq_datasets):
    cos_between_embeds = cos_metric(clean_embeddings, obfuscated_embeddings).mean()
    top_1_intersection = topk_intersection_metric(clean_logits, obfuscated_logits, attention_mask, k=1).mean()
    top_5_intersection = topk_intersection_metric(clean_logits, obfuscated_logits, attention_mask, k=5).mean()

    # nn_fr = nn_fr_metric(obfuscated_embeddings, llm.model.embed_tokens, input_ids, attention_mask)
    
    mean_logvar = logvar.mean()
    mean_mu = mu.mean()

    # reconstruction_rank = reconstruction_rank_metric(obfuscated_embeddings, clean_embeddings, input_ids, attention_mask)

    return {
        "cos": cos_between_embeds,
        "top_1_intersection": top_1_intersection,
        "top_5_intersection": top_5_intersection,
        # "nn_fr": nn_fr,
        # "reconstruction_rank": reconstruction_rank,
        "logvar": mean_logvar,
        "mu": mean_mu
    }
    

def run_epoch(dataloader, llm, tokenizer, sgt, sgt_loss, optimizer, scaler, grad_accumulation_steps, enable_amp, device, mcq_datasets, utility_baseline, do_backprop=True, apply_gradient_clipping=True, log_grad_norms=False):

    losses = defaultdict(list)
    metrics = defaultdict(list)
    gradient_norms = defaultdict(list)

    for batch_idx, batch in enumerate(dataloader):
        x_input_ids, x, x_independent, attention_mask = embed_and_split_batch(llm, batch, device)
        
        with autocast('cuda', dtype=torch.float16, enabled=enable_amp):
            x_tilde, mu, logvar = sgt.sample(x, attention_mask=attention_mask)

            with torch.no_grad():
                mu_independent, logvar_independent = sgt(x_independent, attention_mask=attention_mask)
            
            with torch.no_grad():
                logits_clean = llm(inputs_embeds=x, attention_mask=attention_mask).logits.detach()

            logits_obf = llm(inputs_embeds=x_tilde, attention_mask=attention_mask).logits
            
            loss_dict = sgt_loss(
                x=x, 
                x_tilde=x_tilde, 
                x_independent=x_independent,
                mu=mu, 
                logvar=logvar, 
                mu_independent=mu_independent,
                logvar_independent=logvar_independent,
                logits_clean=logits_clean, 
                logits_obf=logits_obf,
                attention_mask=attention_mask
            )

            metrics_dict = compute_batch_metrics(llm, tokenizer, sgt, x_input_ids, x, mu, logvar, x_tilde, attention_mask, logits_clean, logits_obf, device, utility_baseline)

            for k, v in loss_dict.items():
                losses[k].append(v.detach().cpu().item())

            for k, v in metrics_dict.items():
                metrics[k].append(v.detach().cpu().item())
        
        if do_backprop:
            loss = loss_dict['total_loss'] / grad_accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)

                if log_grad_norms:
                    grad_norm_total = compute_gradient_norm(sgt)
                    gradient_norms['total_grad_norm'].append(grad_norm_total)
                    
                    grad_norms_per_layer = compute_gradient_norm_per_layer(sgt)
                    for name, norm in grad_norms_per_layer.items():
                        gradient_norms[name].append(norm)

                if apply_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(sgt.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    if not do_backprop:
        sep_metrics = compute_separate_metrics(llm, tokenizer, sgt, device, mcq_datasets, utility_baseline)
        for k, v in sep_metrics.items():
            metrics[k] = [v]
    for k, v in losses.items():
        losses[k] = sum(losses[k]) / len(losses[k])

    for k, v in metrics.items():
        metrics[k] = sum(metrics[k]) / len(metrics[k])

    for k, v in gradient_norms.items():
        if v:
            gradient_norms[k] = sum(v) / len(v)

    return losses, metrics, gradient_norms


def demostrate(llm, tokenizer, sgt, neptune_run, epoch, device, prompt='Yo', max_new_tokens=20):
    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        orig_output = llm.generate(inputs["input_ids"], attention_mask=inputs['attention_mask'], max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
        orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
        
        original_embeds = llm.get_input_embeddings()(inputs["input_ids"])
        obfuscated_embeds, _, _ = sgt.sample(original_embeds, attention_mask=inputs['attention_mask'])
        
        obf_output = llm.generate(inputs_embeds=obfuscated_embeds, attention_mask=inputs['attention_mask'], max_new_tokens=max_new_tokens, input_ids=inputs["input_ids"], do_sample=False, temperature=None, top_p=None, top_k=None)
        obf_text = tokenizer.decode(obf_output[0], skip_special_tokens=True)
        
        print(f"Original:   {orig_text}")
        print(f"Obfuscated: {obf_text}")

        neptune_run[f"demo/original_text"].append(orig_text, step=epoch)
        neptune_run[f"demo/obfuscated_text"].append(obf_text, step=epoch)
        neptune_run[f"demo/prompt"].append(prompt, step=epoch)
        
        comparison = f"Prompt: {prompt}\nOriginal: {orig_text}\nObfuscated: {obf_text}"
        neptune_run[f"demo/comparison"].append(comparison, step=epoch)


def push_stats(stats, prefix, neptune_run, epoch):
    for k, v in stats.items():
        neptune_run[f"{prefix}/{k}"].append(v, step=epoch)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)

    run = neptune.init_run(
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
    )

    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    print(json.dumps(config_dict, indent=4))

    run["config"] = config_dict

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_name)
    tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(cfg.model.llm_name, device_map=device)
    llm._dynamo_compile = False 
    for p in llm.parameters():
        p.requires_grad_(False)


    dataset = datasets.load_dataset('ag_news')
    train_texts = [item for item in dataset['train'][:400]['text']]

    eval_texts = [item for item in dataset['train'][800:1000]['text']]

    train_dataset = SGTDataset(train_texts, tokenizer)
    eval_dataset = SGTDataset(eval_texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.train_batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.training.eval_batch_size, shuffle=True, drop_last=True)

    sgt = SGTModel(
        d=llm.config.hidden_size, 
        nhead=cfg.model.sgt.nhead, 
        ff=cfg.model.sgt.ff, 
        layers=cfg.model.sgt.layers,
        mu_init_weight=cfg.model.sgt.mu_init_weight,
        mu_init_bias=cfg.model.sgt.mu_init_bias,
        logvar_init_weight=cfg.model.sgt.logvar_init_weight,
        logvar_init_bias=cfg.model.sgt.logvar_init_bias,
    ).to(device)

    sgt_loss = SGTLoss(
        embedding_weights=llm.model.embed_tokens.weight,

        alpha_mi=cfg.loss.alpha_mi,
        alpha_abs_cos=cfg.loss.alpha_abs_cos,
        alpha_norm=cfg.loss.alpha_norm
    )

    # this weight decay can influence the training
    optimizer = torch.optim.AdamW(sgt.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scaler = GradScaler('cuda')

    mcq_datasets = [
        load_dataset("ai2_arc", "ARC-Challenge")['train'].select(range(100))
    ]

    # initial_utility_obfuscated = evaluate_utility(llm, tokenizer, mcq_datasets, device, sgt=sgt)
    utility_baseline = evaluate_utility(llm, tokenizer, mcq_datasets, device)

    print(f"Initial utility of raw LLM: \t {utility_baseline:.5f}")
    # print(f"Initial utility of obfuscated LLM: \t {initial_utility_obfuscated:.5f}")

    for epoch in tqdm(range(cfg.training.num_epochs)):
        epoch_loss = 0

        sgt.train()

        train_losses, train_metrics, train_grad_norms = run_epoch(train_loader, llm, tokenizer, sgt, sgt_loss, optimizer, scaler, grad_accumulation_steps=cfg.training.grad_accumulation_steps, enable_amp=cfg.training.enable_amp, device=device, mcq_datasets=mcq_datasets, utility_baseline=utility_baseline, do_backprop=True, apply_gradient_clipping=cfg.training.apply_gradient_clipping, log_grad_norms=True)

        push_stats(train_losses, 'train/loss', run, epoch)
        push_stats(train_metrics, 'train/metric', run, epoch)
        push_stats(train_grad_norms, 'train/gradients', run, epoch) 

        if epoch % cfg.training.eval_frequency == 0:
            sgt.eval()
            with torch.inference_mode():
                eval_losses, eval_metrics, _ = run_epoch(eval_loader, llm, tokenizer, sgt, sgt_loss, optimizer, scaler, grad_accumulation_steps=cfg.training.grad_accumulation_steps, enable_amp=cfg.training.enable_amp, device=device, mcq_datasets=mcq_datasets, utility_baseline=utility_baseline, do_backprop=False, apply_gradient_clipping=cfg.training.apply_gradient_clipping, log_grad_norms=False)

            push_stats(eval_losses, 'eval/loss', run, epoch)
            push_stats(eval_metrics, 'eval/metric', run, epoch)


        print(f"Epoch #{epoch}")
        print(f"Train: loss = {train_losses['total_loss']:.4f}. \t Eval: loss = {eval_losses['total_loss']:.4f}")
        if epoch % cfg.training.demonstration_frequency == 0:
            demostrate(llm, tokenizer, sgt, run, epoch, device, prompt="This is translation of 'The quick brown fox jumps' to French:", max_new_tokens=20)


    run.stop()


if __name__ == "__main__":
    main()