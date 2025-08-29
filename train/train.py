import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.amp import GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
import datasets
import neptune
from loss_practical import SGTLossPractical

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from metrics import cos_metric, topk_intersection_metric, reconstruction_rank_metric, evaluate_utility, nn_fr_metric, mrp_fr_metric, ttr_k_metric, sym_ttr_k_metric

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


def compute_l2_per_layer(model): 
    weight_norms = {}
    for name, param in model.named_parameters():
        if 'logvar_head' in name:
            if param.grad is not None:
                grad_norm = param.detach().pow(2).sum().sqrt().item()
                weight_norms[f"weight_norm/{name}"] = grad_norm

    return weight_norms


def embed_batch(llm, batch, device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    clean_embeds = None

    if 'clean_embeds' in batch:
        clean_embeds = batch['clean_embeds'].to(device)
    
    with torch.no_grad():
        embeds = llm.get_input_embeddings()(input_ids)

    x = embeds
    x_mask = attention_mask

    return input_ids, x, x_mask, clean_embeds


def compute_separate_metrics(llm, tokenizer, sgt, device, mcq_datasets, utility_baseline):
    utility = evaluate_utility(llm, tokenizer, mcq_datasets, device, sgt=sgt)

    return {
        "utility": utility - utility_baseline
    }

def compute_batch_metrics(llm, tokenizer, sgt, input_ids, clean_embeddings, mu, logvar, obfuscated_embeddings, attention_mask, clean_logits, obfuscated_logits, device, mcq_datasets):
    cos_between_embeds = cos_metric(clean_embeddings, obfuscated_embeddings).mean()
    top_1_intersection = topk_intersection_metric(clean_logits, obfuscated_logits, attention_mask, k=1).mean()
    top_5_intersection = topk_intersection_metric(clean_logits, obfuscated_logits, attention_mask, k=5).mean()

    nn_fr = nn_fr_metric(obfuscated_embeddings, llm.model.embed_tokens, input_ids, attention_mask)

    reconstruction_rank = reconstruction_rank_metric(obfuscated_embeddings, llm.model.embed_tokens, input_ids, attention_mask)

    mrp_fr = mrp_fr_metric(obfuscated_embeddings, llm.model.embed_tokens, input_ids, attention_mask, r=1)

    ttr_k = ttr_k_metric(obfuscated_embeddings, llm.model.embed_tokens, input_ids, attention_mask, k=10)

    sym_ttr_k = sym_ttr_k_metric(obfuscated_embeddings, llm.model.embed_tokens, input_ids, attention_mask, k=10)
    
    mean_logvar = logvar.mean()
    mean_mu = mu.mean()

    return {
        "cos": cos_between_embeds,
        "top_1_intersection": top_1_intersection,
        "top_5_intersection": top_5_intersection,
        "nn_fr": nn_fr,
        "reconstruction_rank": reconstruction_rank,
        "mrp_fr": mrp_fr,
        "ttr_k": ttr_k,
        "sym_ttr_k": sym_ttr_k,
        "logvar": mean_logvar,
        "mu": mean_mu
    }
    

def run_epoch(dataloader, llm, tokenizer, sgt, sgt_loss, optimizer, scaler, grad_accumulation_steps, enable_amp, device, mcq_datasets, utility_baseline, 
              do_backprop=True, apply_gradient_clipping=True, log_grad_norms=False):

    losses = defaultdict(list)
    metrics = defaultdict(list)
    gradient_norms = defaultdict(list)
    l2_norms = defaultdict(list)

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        x_input_ids, x, attention_mask, logits_clean = embed_batch(llm, batch, device)
        
        x_tilde, mu, logvar = sgt.sample(x, attention_mask=attention_mask)
        
        if logits_clean is None:
            with torch.no_grad():
                logits_clean = llm(inputs_embeds=x, attention_mask=attention_mask).logits.detach()

        logits_obf = llm(inputs_embeds=x_tilde, attention_mask=attention_mask).logits
        
        loss_dict = sgt_loss(
            x=x, 
            x_tilde=x_tilde, 
            mu=mu, 
            logvar=logvar, 
            logits_clean=logits_clean, 
            logits_obf=logits_obf,
            attention_mask=attention_mask,
        )

        with torch.inference_mode():
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


                    l2_norms_per_layer = compute_l2_per_layer(sgt)
                    for name, norm in l2_norms_per_layer.items():
                        l2_norms[name].append(norm)
                    

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

    for k, v in l2_norms.items():
        if v:
            l2_norms[k] = sum(v) / len(v)

    return losses, metrics, gradient_norms, l2_norms


def demostrate(llm, tokenizer, sgt, neptune_run, epoch, device, prompt='Yo', max_new_tokens=20):
    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        orig_output = llm.generate(inputs["input_ids"], attention_mask=inputs['attention_mask'], max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
        orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
        
        original_embeds = llm.get_input_embeddings()(inputs["input_ids"])
        obfuscated_embeds, _, _ = sgt.sample(original_embeds, attention_mask=inputs['attention_mask'])

        obf_output = llm.generate(inputs_embeds=obfuscated_embeds, attention_mask=inputs['attention_mask'], max_new_tokens=max_new_tokens, input_ids=inputs["input_ids"], do_sample=False, temperature=None, top_p=None, top_k=None)
        obf_text = tokenizer.decode(obf_output[0], skip_special_tokens=True)
        
        embedding_matrix = llm.get_input_embeddings().weight
        inverted_tokens = []
        for pos in range(obfuscated_embeds.size(1)):
            similarities = torch.cosine_similarity(obfuscated_embeds[0, pos:pos+1], embedding_matrix, dim=1)
            inverted_tokens.append(similarities.argmax().item())
        inverted_text = tokenizer.decode(inverted_tokens, skip_special_tokens=True)
        
        avg_cos = torch.cosine_similarity(original_embeds[0], obfuscated_embeds[0], dim=1).mean().item()
        
        print(f"Original:   {orig_text}")
        print(f"Obfuscated: {obf_text}")
        print(f"Inverted:   {inverted_text}")
        print(f"Avg cos similarity (orig vs obf): {avg_cos:.4f}")

        neptune_run[f"demo/original_text"].append(orig_text, step=epoch)
        neptune_run[f"demo/obfuscated_text"].append(obf_text, step=epoch)
        neptune_run[f"demo/inverted_text"].append(inverted_text, step=epoch)
        neptune_run[f"demo/prompt"].append(prompt, step=epoch)
        
        comparison = f"Prompt: {prompt}\nOriginal: {orig_text}\nObfuscated: {obf_text}\nInverted: {inverted_text}"
        neptune_run[f"demo/comparison"].append(comparison, step=epoch)


def push_stats(stats, prefix, neptune_run, epoch):
    for k, v in stats.items():
        neptune_run[f"{prefix}/{k}"].append(v, step=epoch)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    checkpoint_path = f'./checkpoints/{cfg.name.replace(".", "_")}'

    os.makedirs(checkpoint_path, exist_ok=True)

    should_cache_clean_logits = cfg.model.cache_clean_logits

    device = torch.device(cfg.device)
    best_eval_loss = float('inf')

    run = neptune.init_run(
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
    )

    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    print(json.dumps(config_dict, indent=4))

    run["config"] = config_dict

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_name)
    # tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(cfg.model.llm_name, device_map=device)
    llm.generation_config.pad_token_id = tokenizer.pad_token_id

    for p in llm.parameters():
        p.requires_grad_(False)

    dataset = datasets.load_dataset(cfg.data.dataset_name)
    
    train_texts = [item for item in dataset['train'][:cfg.data.num_samples]['text']]

    eval_texts = [item for item in dataset['train'][cfg.data.num_samples:cfg.data.num_samples + 400]['text']]

    llm_for_embeds_clean_precompute = None
    if should_cache_clean_logits:
        llm_for_embeds_clean_precompute = llm

    train_dataset = SGTDataset(train_texts, tokenizer, llm_for_embeds_clean_precompute=llm_for_embeds_clean_precompute, max_length=cfg.model.tokenizer_max_length)
    eval_dataset = SGTDataset(eval_texts, tokenizer, llm_for_embeds_clean_precompute=llm_for_embeds_clean_precompute, max_length=cfg.model.tokenizer_max_length)
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


    sgt_loss = SGTLossPractical(
        embedding_weights=llm.model.embed_tokens.weight,
        alpha_utility=cfg.loss.alpha_utility,
        alpha_obfuscation=cfg.loss.alpha_obfuscation,
        alpha_abs_cos=cfg.loss.alpha_abs_cos,
        alpha_logvar_mse=cfg.loss.alpha_logvar_mse,
        alpha_norm=cfg.loss.alpha_norm,
    )

    decay, no_decay = [], []
    for n,p in sgt.named_parameters():
        if any(k in n for k in ['mu_head.', 'logvar_head.']):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': cfg.training.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}],
        lr=cfg.training.lr, betas=(0.9, 0.95), eps=1e-5
    )
    
    scaler = GradScaler('cuda')

    mcq_datasets = [
        load_dataset("ai2_arc", "ARC-Challenge")['train'].select(range(100))
    ]

    initial_utility_obfuscated = evaluate_utility(llm, tokenizer, mcq_datasets, device, sgt=sgt)
    utility_baseline = evaluate_utility(llm, tokenizer, mcq_datasets, device)

    print(f"Initial utility of obfuscated LLM: \t {initial_utility_obfuscated:.5f}")
    print(f"Initial utility of normal LLM: \t {utility_baseline:.5f}")

    for epoch in tqdm(range(cfg.training.num_epochs)):
        epoch_loss = 0

        sgt.train()

        train_losses, train_metrics, train_grad_norms, l2_norms = run_epoch(train_loader, llm, tokenizer, sgt, sgt_loss, optimizer, scaler, grad_accumulation_steps=cfg.training.grad_accumulation_steps, enable_amp=cfg.training.enable_amp, device=device, mcq_datasets=mcq_datasets, utility_baseline=utility_baseline, do_backprop=True, apply_gradient_clipping=cfg.training.apply_gradient_clipping, log_grad_norms=True)

        push_stats(train_losses, 'train/loss', run, epoch)
        push_stats(train_metrics, 'train/metric', run, epoch)
        push_stats(train_grad_norms, 'train/gradients', run, epoch)
        push_stats(l2_norms, 'train/l2_norms', run, epoch) 

        if epoch % cfg.training.eval_frequency == 0 and epoch != 0:
            sgt.eval()
            with torch.inference_mode():
                eval_losses, eval_metrics, _, _ = run_epoch(eval_loader, llm, tokenizer, sgt, sgt_loss, optimizer, scaler, grad_accumulation_steps=cfg.training.grad_accumulation_steps, enable_amp=cfg.training.enable_amp, device=device, mcq_datasets=mcq_datasets, utility_baseline=utility_baseline, do_backprop=False, apply_gradient_clipping=cfg.training.apply_gradient_clipping, log_grad_norms=False)

            push_stats(eval_losses, 'eval/loss', run, epoch)
            push_stats(eval_metrics, 'eval/metric', run, epoch)

            if eval_losses['total_loss'] < best_eval_loss:
                best_eval_loss = eval_losses['total_loss']
                print(f"Saving new best model with eval loss = {best_eval_loss}")
                torch.save(sgt.state_dict(), f'{checkpoint_path}/best_sgt.pt')

            print(f"Eval: loss = {eval_losses['total_loss']:.4f}, utility = {eval_metrics['utility']}")

        print(f"Epoch #{epoch}")

        print(f"Train: total loss = {train_losses['total_loss']:.4f}, ",
            f"utility loss = {train_losses['utility']:.4f}, ",
            f"logvar_mse = {train_losses['logvar_mse']:.4f}, ",
            f"abs cos loss = {train_losses['abs_cos']:.4f}.",
            )
            
        if epoch % cfg.training.demonstration_frequency == 0:
            demostrate(llm, tokenizer, sgt, run, epoch, device, prompt="This is translation of 'The quick brown fox jumps' to Russian:", max_new_tokens=20)


    run.stop()


if __name__ == "__main__":
    main()