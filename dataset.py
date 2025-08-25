import torch

from tqdm.auto import tqdm

from collections import defaultdict

from torch.utils.data import Dataset

class SGTDataset(Dataset):
    def __init__(self, texts, tokenizer, llm_for_embeds_clean_precompute=None, precompution_batch_size=32, max_length=16):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', 
                                 max_length=max_length, return_tensors='pt')
        
        self.llm_for_embeds_clean_precompute = llm_for_embeds_clean_precompute
        self.precompution_batch_size = precompution_batch_size

        self.cached_clean_embeds = {}

        if llm_for_embeds_clean_precompute is not None:
            device = next(llm_for_embeds_clean_precompute.parameters()).device

            for start_idx in tqdm(range(0, len(self), precompution_batch_size), desc='Precomputing logits'):

                inputs = {key: val[start_idx: start_idx + precompution_batch_size] for key, val in self.encodings.items()}
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    logits = llm_for_embeds_clean_precompute(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits.detach().cpu()

                    for i in range(logits.shape[0]):
                        self.cached_clean_embeds[start_idx + i] = logits[i, :]
    

    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        outputs = {key: val[idx] for key, val in self.encodings.items()}
        if idx in self.cached_clean_embeds:
            outputs['clean_embeds'] = self.cached_clean_embeds[idx]
        return outputs
