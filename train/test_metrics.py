import pytest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import _reconstruction_ranks, mrp_fr_metric, nn_fr_metric as nn_fr, sym_ttr_k_metric, ttr_k_metric

# device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to(device)

### ranks


@pytest.mark.parametrize("similarity", ["euclid", "cosine"])
def test_reconstruction_ranks_zero(similarity):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)

    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    ranks = _reconstruction_ranks(
        clean_embeddings,
        llm.model.embed_tokens.weight.to(device),
        tokenized["input_ids"],
        tokenized["attention_mask"],
        similarity=similarity,
    )

    # check size
    assert ranks.numel() == tokenized["attention_mask"].sum().item()

    # check equal to 0
    assert torch.all(ranks == 0)


### nn_fr


@pytest.mark.parametrize("similarity", ["euclid", "cosine"])
def test_nn_fr_zero(similarity):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)

    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    actual = nn_fr(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        similarity=similarity,
    )

    assert actual == 0


@pytest.mark.parametrize("similarity", ["euclid", "cosine"])
def test_nn_fr_one_embed_wrong(similarity):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)

    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    clean_embeddings[0][1] += torch.randn_like(clean_embeddings[0][1])

    actual = nn_fr(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        similarity=similarity,
    )

    assert actual == 100.0 / 9


@pytest.mark.parametrize("similarity", ["euclid", "cosine"])
def test_nn_fr_one_embed_wrong_with_attention_mask(similarity):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)

    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    clean_embeddings[0][1] += torch.randn_like(clean_embeddings[0][1])

    tokenized["attention_mask"][0][1] = 0

    actual = nn_fr(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        similarity=similarity,
    )

    assert actual == 0


### mrp_fr


@pytest.mark.parametrize("similarity,r", [("euclid", 1), ("cosine", 1), ("euclid", 5)])
def test_mrp_fr_real(similarity, r):
    """
    Using real embeddings
     -> r-th farthest != gt
    """
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)

    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    actual = mrp_fr_metric(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        r,
        similarity,
    )

    assert actual == 100.0


@pytest.mark.parametrize("similarity", (["euclid", "cosine"]))
def test_mrp_fr_reverse(similarity):
    """
    Using real embeddings
     -> r-th farthest != gt
    """
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)

    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    actual = mrp_fr_metric(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        llm.model.embed_tokens.weight.size(0),
        similarity,
    )

    assert actual == 0.0


### ttr_k


@pytest.mark.parametrize("similarity,k", [("euclid", 1), ("cosine", 1), ("euclid", 5)])
def test_ttr_k_real(similarity, k):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    actual = ttr_k_metric(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        k,
        similarity,
    )
    assert actual == 0


@pytest.mark.parametrize("similarity,k", [("euclid", 1), ("cosine", 1), ("euclid", 5)])
def test_ttr_k_one_embed_wrong(similarity, k):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    clean_embeddings[0][1] += torch.randn_like(clean_embeddings[0][1])

    actual = ttr_k_metric(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        k,
        similarity,
    )
    assert actual == 100.0 / 9


@pytest.mark.parametrize("similarity,k", [("euclid", 1)])
def test_ttr_k_one_embed_wrong_with_mask(similarity, k):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    clean_embeddings[0][1] += torch.randn_like(clean_embeddings[0][1])
    tokenized["attention_mask"][0][1] = 0

    actual = ttr_k_metric(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        k,
        similarity,
    )
    assert actual == 0.0


### sym_ttr_k


@pytest.mark.parametrize("similarity,k", [("euclid", 1), ("cosine", 1), ("euclid", 5)])
def test_sym_ttr_k_real(similarity, k):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    actual = sym_ttr_k_metric(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        k,
        similarity,
    )
    assert actual == 0


@pytest.mark.parametrize("similarity,k", [("euclid", 1), ("cosine", 1), ("euclid", 5)])
def test_sym_ttr_k_one_embed_wrong(similarity, k):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    clean_embeddings[0][1] += torch.randn_like(clean_embeddings[0][1])

    actual = sym_ttr_k_metric(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        k,
        similarity,
    )
    assert actual == 100.0 / 9


@pytest.mark.parametrize("similarity,k", [("euclid", 1)])
def test_sym_ttr_k_one_embed_wrong_with_mask(similarity, k):
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors="pt").to(device)
    clean_embeddings = llm.model.embed_tokens(tokenized["input_ids"])

    clean_embeddings[0][1] += torch.randn_like(clean_embeddings[0][1])
    tokenized["attention_mask"][0][1] = 0

    actual = sym_ttr_k_metric(
        clean_embeddings,
        llm.model.embed_tokens,
        tokenized["input_ids"],
        tokenized["attention_mask"],
        k,
        similarity,
    )
    assert actual == 0.0
