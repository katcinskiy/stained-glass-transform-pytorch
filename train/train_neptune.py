import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
import datasets
import neptune
from train.loss_paper import SGTLoss

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from metrics import cos_metric, topk_intersection_metric, reconstruction_rank_metric

from collections import defaultdict

from dataset import SGTDataset

from sgt_model import SGTModel




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


def compute_metrics(input_ids, clean_embeddings, obfuscated_embeddings, attention_mask, clean_logits, obfuscated_logits):
    cos_between_embeds = cos_metric(clean_embeddings, obfuscated_embeddings).mean()
    top_1_intersection = topk_intersection_metric(clean_logits, obfuscated_logits, attention_mask, k=1).mean()
    top_5_intersection = topk_intersection_metric(clean_logits, obfuscated_logits, attention_mask, k=5).mean()

    # reconstruction_rank = reconstruction_rank_metric(obfuscated_embeddings, clean_embeddings, input_ids, attention_mask)

    return {
        "cos": cos_between_embeds,
        "top_1_intersection": top_1_intersection,
        "top_5_intersection": top_5_intersection,
        # "reconstruction_rank": reconstruction_rank
    }
    

def run_epoch(dataloader, llm, sgt, sgt_loss, optimizer, scaler, grad_accumulation_steps, enable_amp, device, do_backprop=True, apply_gradient_clipping=True):

    losses = defaultdict(list)
    metrics = defaultdict(list)

    for batch_idx, batch in enumerate(dataloader):
        x_input_ids, x, x_independent, attention_mask = embed_and_split_batch(llm, batch, device)
        
        with autocast('cuda', dtype=torch.float16, enabled=enable_amp):
            x_tilde, mu, logvar = sgt.sample(x)
            mu_independent, logvar_independent = sgt(x_independent)
            
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

            metrics_dict = compute_metrics(x_input_ids, x, x_tilde, attention_mask, logits_clean, logits_obf)

            for k, v in loss_dict.items():
                losses[k].append(v.detach().cpu().item())

            for k, v in metrics_dict.items():
                metrics[k].append(v.detach().cpu().item())
        
        if do_backprop:
            loss = loss_dict['total_loss'] / grad_accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)

                if apply_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(sgt.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
    for k, v in losses.items():
        losses[k] = sum(losses[k]) / len(losses[k])

    for k, v in metrics.items():
        metrics[k] = sum(metrics[k]) / len(metrics[k])

    return losses, metrics


def demostrate(llm, tokenizer, sgt, neptune_run, epoch, device, prompt='Yo', max_new_tokens=20):
    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        orig_output = llm.generate(inputs["input_ids"], attention_mask=inputs['attention_mask'], max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
        orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
        
        original_embeds = llm.get_input_embeddings()(inputs["input_ids"])
        obfuscated_embeds, _, _ = sgt.sample(original_embeds)
        
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
    for p in llm.parameters():
        p.requires_grad_(False)


    dataset = datasets.load_dataset('ag_news')
    train_texts = [item for item in dataset['train'][:400]['text']]

    eval_texts = [item for item in dataset['train'][800:816 + 1]['text']]


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

    for epoch in tqdm(range(cfg.training.num_epochs)):
        epoch_loss = 0

        sgt.train()

        train_losses, train_metrics = run_epoch(train_loader, llm, sgt, sgt_loss, optimizer, scaler, grad_accumulation_steps=cfg.training.grad_accumulation_steps, enable_amp=cfg.training.enable_amp, device=device, do_backprop=True, apply_gradient_clipping=cfg.training.apply_gradient_clipping)

        push_stats(train_losses, 'train/loss', run, epoch)
        push_stats(train_metrics, 'train/metric', run, epoch)

        if epoch % cfg.training.eval_frequency == 0:
            sgt.eval()
            with torch.inference_mode():
                eval_losses, eval_metrics = run_epoch(eval_loader, llm, sgt, sgt_loss, optimizer, scaler, grad_accumulation_steps=cfg.training.grad_accumulation_steps, enable_amp=cfg.training.enable_amp, device=device, do_backprop=False, apply_gradient_clipping=cfg.training.apply_gradient_clipping)

            push_stats(eval_losses, 'eval/loss', run, epoch)
            push_stats(eval_metrics, 'eval/metric', run, epoch)



        print(f"Epoch #{epoch}")
        print(f"Train: loss = {train_losses['total_loss']:.4f}. \t Eval: loss = {eval_losses['total_loss']:.4f}")
        if epoch % cfg.training.demonstration_frequency == 0:
            demostrate(llm, tokenizer, sgt, run, epoch, device, prompt='Hello my friend', max_new_tokens=20)


    run.stop()


if __name__ == "__main__":
    main()