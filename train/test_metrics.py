import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import nn_fr_metric as nn_fr

device = torch.device('cuda:0')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to(device)

def test_nn_fr_zero():
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors='pt').to(device)

    clean_embeddings = llm.model.embed_tokens(tokenized['input_ids'])

    actual = nn_fr(clean_embeddings, llm.model.embed_tokens, tokenized['input_ids'], tokenized['attention_mask'])

    assert actual == 0

def test_nn_fr_one_embed_wrong():
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors='pt').to(device)

    clean_embeddings = llm.model.embed_tokens(tokenized['input_ids'])

    clean_embeddings[0][1] += torch.randn_like(clean_embeddings[0][1])

    actual = nn_fr(clean_embeddings, llm.model.embed_tokens, tokenized['input_ids'], tokenized['attention_mask'])

    assert actual == 100.0 / 9

def test_nn_fr_one_embed_wrong_with_attention_mask():
    text = "The quick brown fox jumps over the lazy dog"
    tokenized = tokenizer(text, return_tensors='pt').to(device)

    clean_embeddings = llm.model.embed_tokens(tokenized['input_ids'])

    clean_embeddings[0][1] += torch.randn_like(clean_embeddings[0][1])

    tokenized['attention_mask'][0][1] = 0

    actual = nn_fr(clean_embeddings, llm.model.embed_tokens, tokenized['input_ids'], tokenized['attention_mask'])

    assert actual == 0

