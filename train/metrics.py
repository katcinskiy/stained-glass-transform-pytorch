import re
import torch

import torch.nn.functional as F

from tqdm.auto import tqdm

def cos_metric(x, y):
    cos_sim = F.cosine_similarity(x, y, dim=-1)
    return cos_sim

def topk_intersection_metric(logits1, logits2, attention_mask, k):

    topk_1 = logits1.topk(k, dim=-1)[1]
    topk_2 = logits2.topk(k, dim=-1)[1]
    
    intersection_size = torch.zeros_like(topk_1[..., 0])

    for i in range(k):
        for j in range(k):
            intersection_size += (topk_1[..., i] == topk_2[..., j])
    
    intersection_size = intersection_size.float() * attention_mask
    return intersection_size.sum(-1) / attention_mask.sum(-1) / k

def nn_fr_metric(obf_embeds, embedding_layer, input_ids, attention_mask):
    obf_embeds_normalized = F.normalize(obf_embeds.cpu(), dim=-1).cpu() #TODO: remove cpu if enougth memory
    embedding_layer_normalized = F.normalize(embedding_layer.weight.cpu()) #TODO: remove cpu if enougth memory

    result = obf_embeds_normalized @ embedding_layer_normalized.T

    result = result.argmax(dim=-1)

    fail = (result != input_ids.cpu()) & attention_mask.cpu()

    return 100 * fail.sum() / attention_mask.cpu().sum()

def reconstruction_rank_metric(obf_embeds, clean_embeds, input_ids, mask):
    valid_mask = mask.flatten() == 1
    obf_flat = obf_embeds.view(-1, obf_embeds.size(-1))[valid_mask]
    ids_flat = input_ids.flatten()[valid_mask]
    
    distances = torch.cdist(obf_flat, clean_embeds)
    ranks = (torch.argsort(distances, dim=1) == ids_flat.unsqueeze(1)).nonzero()[:, 1] + 1
    
    return ranks.float().mean()

def evaluate_utility(llm, tokenizer, mcq_datasets, device, sgt=None):
    mcqs = []

    for dataset in mcq_datasets:
        mcqs.append(evaluate_mcq(llm, tokenizer, dataset, device, sgt=sgt))

    return torch.tensor(sum(mcqs) / len(mcqs))

def evaluate_mcq(llm, tokenizer, dataset, device, sgt=None):
    correct = 0
    
    for example in tqdm(dataset, desc='Evaluating MCQ'):
        choices = "\n".join([f"{l}: {t}" for l, t in zip(example['choices']['label'], example['choices']['text'])])
        prompt = f"Question: {example['question']}\n\nChoices:\n{choices}\n\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            if sgt is not None:
                clean_embeddings = llm.model.embed_tokens(inputs['input_ids'])
                obf_embeddings, _, _ = sgt.sample(clean_embeddings, attention_mask=inputs['attention_mask'])
                output = llm.generate(inputs_embeds=obf_embeddings, attention_mask=inputs['attention_mask'], max_new_tokens=10, input_ids=inputs["input_ids"])
            else:
                output = llm.generate(**inputs, max_new_tokens=10)

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = re.search(r'Answer:\s*([A-E])', response)    
    
        if pred and pred.group(1) == example['answerKey']:
            correct += 1
    
    return correct / len(dataset)


def mrp_fr_metric():
    pass
    # your impl here

def ttr_k_metric():
    pass
    # your impl here

def sym_ttr_k_metric():
    pass
    # your impl here