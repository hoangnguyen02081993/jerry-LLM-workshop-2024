import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from torch.utils.data import Dataset
import os
from lib.model import IModel
import torch.mps

# The data set format is a list of tuples (text, label)
class ScamSMSDatasetV1(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.data = []
        
        for _, d in enumerate(data):
            encoding = self.tokenizer(d['text'], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            self.data.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'label': d['label']
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data['input_ids'], data['attention_mask'], data['label']

class ScamSMSClassifierV2(IModel):
    def __init__(self, model_name, num_classes, dropout):
        super(ScamSMSClassifierV2, self).__init__()

        if not os.path.exists(f"data/tokenizer"):
            os.makedirs(f"data/tokenizer")

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, cache_dir=f"data/tokenizer/{model_name}")

        self.bert = XLMRobertaModel.from_pretrained(model_name)
        
        # Fully Connected Layer (Converts BERT output to class logits)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Dropout (Prevents overfitting)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        # Get BERT's output
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token (first token of each sequence)
        cls_output = bert_output.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

        # Apply dropout
        cls_output = self.dropout(cls_output)

        # Fully connected layer for classification
        logits = self.fc(cls_output)  # Shape: (batch_size, num_classes)

        return logits

    def encode(self, text, max_length=256):
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        return encoding['input_ids'], encoding['attention_mask']
    
    def decode(self, tokens):
        return self.tokenizer(tokens)

    def create_data_set(self, data):
        return ScamSMSDatasetV1(data, self.tokenizer)
    
    def start_train(self, data, lr=0.0001, epochs=10):
        train_data_set = self.create_data_set(data)
        train_loader = DataLoader(train_data_set, batch_size=1)

        print("Training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        self.train()
        for i in range(epochs):
            step = 1
            total_step = len(train_loader.dataset)
            for input_ids, attention_mask, labels in train_loader:
                input_ids.to(self.device)
                attention_mask.to(self.device)
                optimizer.zero_grad()
                logits = self(input_ids[0], attention_mask[0])
                loss = criterion(logits, labels.to(self.device)).to(self.device)
                loss.backward()
                optimizer.step()

                print(f"[Epoch {i+1}/{epochs}] Step {step}/{total_step} completed")
                step += 1

                # total_loss += loss.item()
            print(f"Epoch: {i+1}/{epochs} completed")

    def start_eval(self, data):
        self.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for d in data:
                input_ids, attention_mask, labels = self.tokenizer(d['text'], truncation=True, padding="max_length", max_length=256, return_tensors="pt")['input_ids'], self.tokenizer(d['text'], truncation=True, padding="max_length", max_length=256, return_tensors="pt")
                outputs = self(input_ids, attention_mask)
                total += 1
                _, predicted = torch.max(outputs, 1)
                print(f"With test {d['text']} predicted: {predicted} actual: {labels}")
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {correct / total  * 100}%")

        