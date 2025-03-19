from torch.utils.data import Dataset
import torch

# The data set format is a list of tuples (text, label)
class ScamSMSDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.data = []
        
        for _, d in enumerate(data):
            token_ids = self.tokenizer.encode(d['text'])
            chunk = token_ids[max_length - len(token_ids)]
            print(d['text'], chunk)
            self.data.append({
                'text': torch.tensor(chunk),
                'label': d['label']
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data['text'], data['label']