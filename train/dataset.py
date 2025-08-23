from torch.utils.data import Dataset

class SGTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=16):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', 
                                 max_length=max_length, return_tensors='pt')
    
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
