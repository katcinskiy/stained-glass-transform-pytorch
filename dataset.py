import torch

from tqdm.auto import tqdm

from collections import defaultdict

from torch.utils.data import Dataset

class SGTDataset(Dataset):
    def __init__(self, texts, tokenizer, llm_for_embeds_clean_precompute=None, max_length=16):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', 
                                 max_length=max_length, return_tensors='pt')
        
        self.llm_for_embeds_clean_precompute = llm_for_embeds_clean_precompute

        self.cached_clean_embeds = {}
        if llm_for_embeds_clean_precompute is not None:
            device = next(llm_for_embeds_clean_precompute.parameters()).device
            for idx in tqdm(range(len(self)), desc='Precomputing logits'):
                inputs = self[idx]
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    self.cached_clean_embeds[idx] = llm_for_embeds_clean_precompute(input_ids=inputs['input_ids'].unsqueeze(0), attention_mask=inputs['attention_mask'].unsqueeze(0)).logits[0, :].detach().cpu()
    
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        outputs = {key: val[idx] for key, val in self.encodings.items()}
        if idx in self.cached_clean_embeds:
            outputs['clean_embeds'] = self.cached_clean_embeds[idx]
        return outputs
