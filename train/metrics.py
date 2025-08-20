import re
import torch

import torch.nn.functional as F

from tqdm.auto import tqdm


def cos_metric(x, y):
    cos_sim = F.cosine_similarity(x, y, dim=-1)

    return cos_sim


def topk_intersection_metric(logits1, logits2, attention_mask, k):
    topk_1 = logits1.topk(k, dim=-1)[1].unsqueeze(-1)
    topk_2 = logits2.topk(k, dim=-1)[1].unsqueeze(-2)

    intersection_size = (topk_1 == topk_2).sum((-1, -2))
    intersection_size = intersection_size.float() * attention_mask

    return intersection_size.sum(-1) / attention_mask.sum(-1) / k


def _reconstruction_ranks(obf_embeds, vocab_embeds, input_ids, mask, similarity="euclid"):
    valid_mask = mask.flatten() == 1
    obf_flat = obf_embeds.view(-1, obf_embeds.size(-1))[valid_mask]
    ids_flat = input_ids.flatten()[valid_mask]

    if similarity == "euclid":
        distances = torch.cdist(obf_flat, vocab_embeds)
    elif similarity == "cosine":
        obf_flat_norm = F.normalize(obf_flat, p=2, dim=-1)
        vocab_embeds_norm = F.normalize(vocab_embeds, p=2, dim=-1)
        dot_products = obf_flat_norm @ vocab_embeds_norm.T
        distances = 1 - dot_products
    else:
        raise ValueError(f"Unknown similarity type: {similarity}")

    sorted_indices = distances.argsort(dim=1)  # ascending
    ranks = (sorted_indices == ids_flat.unsqueeze(1)).nonzero()[:, 1]

    return ranks


def reconstruction_rank_metric(obf_embeds, embedding_layer, input_ids, mask, similarity="euclid"):
    vocab_embeds = embedding_layer.weight
    return (
        (
            _reconstruction_ranks(
                obf_embeds.cpu(),
                vocab_embeds.cpu(),
                input_ids.cpu(),
                mask.cpu(),
                similarity,
            )
            + 1
        )
        .float()
        .mean()
    )


def nn_fr_metric(obf_embeds, embedding_layer, input_ids, mask, similarity="euclid"):
    vocab_embeds = embedding_layer.weight
    ranks = _reconstruction_ranks(
        obf_embeds.cpu(),
        vocab_embeds.cpu(),
        input_ids.cpu(),
        mask.cpu(),
        similarity,
    )

    fail = ranks != 0

    return 100 * fail.sum() / mask.cpu().sum()


def mrp_fr_metric(obf_embeds, embedding_layer, input_ids, mask, r=1, similarity="euclid"):
    vocab_embeds = embedding_layer.weight
    ranks = _reconstruction_ranks(
        obf_embeds.cpu(),
        vocab_embeds.cpu(),
        input_ids.cpu(),
        mask.cpu(),
        similarity,
    )

    target_rank = vocab_embeds.size(0) - r
    fail = ranks != target_rank

    return 100 * fail.sum() / mask.cpu().sum()


def ttr_k_metric(obf_embeds, embedding_layer, input_ids, mask, k=1, similarity="euclid"):
    vocab_embeds = embedding_layer.weight
    ranks = _reconstruction_ranks(
        obf_embeds.cpu(),
        vocab_embeds.cpu(),
        input_ids.cpu(),
        mask.cpu(),
        similarity,
    )

    fail = ranks >= k

    return 100 * fail.sum() / mask.cpu().sum()


def sym_ttr_k_metric(obf_embeds, embedding_layer, input_ids, mask, k=1, similarity="euclid"):
    vocab_embeds = embedding_layer.weight
    ranks = _reconstruction_ranks(
        obf_embeds.cpu(),
        vocab_embeds.cpu(),
        input_ids.cpu(),
        mask.cpu(),
        similarity,
    )

    fail = (ranks >= k) & (ranks < vocab_embeds.size(0) - k)

    return 100 * fail.sum() / mask.cpu().sum()


def evaluate_utility(llm, tokenizer, mcq_datasets, device, sgt=None):
    mcqs = []

    for dataset in mcq_datasets:
        mcqs.append(evaluate_mcq(llm, tokenizer, dataset, device, sgt=sgt))

    return torch.tensor(sum(mcqs) / len(mcqs))


def evaluate_mcq(llm, tokenizer, dataset, device, sgt=None):
    correct = 0

    for example in tqdm(dataset, desc="Evaluating MCQ"):
        choices = "\n".join([f"{l}: {t}" for l, t in zip(example["choices"]["label"], example["choices"]["text"])])
        prompt = f"Question: {example['question']}\n\nChoices:\n{choices}\n\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            if sgt is not None:
                clean_embeddings = llm.model.embed_tokens(inputs["input_ids"])
                obf_embeddings, _, _ = sgt.sample(clean_embeddings, attention_mask=inputs["attention_mask"])
                output = llm.generate(
                    inputs_embeds=obf_embeddings,
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    input_ids=inputs["input_ids"],
                )
            else:
                output = llm.generate(**inputs, max_new_tokens=10)

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = re.search(r"Answer:\s*([A-E])", response)

        if pred and pred.group(1) == example["answerKey"]:
            correct += 1

    return correct / len(dataset)
