import torch.nn as nn
from torch.utils.data import DataLoader
import tiktoken
import torch
from lib.dataset import ScamSMSDataset

class ScamSMSClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(ScamSMSClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, num_classes)

        # For example
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)  # Average embeddings
        x = self.fc(x)
        x = self.relu(x)
        x = self.output(x)
        return x

    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def set_device(self, device):
        self.to(device)
        self.device = device

    def create_data_set(self, data):
        return ScamSMSDataset(data, self.tokenizer)
    
    def start_train(self, data, epochs=10):
        train_data_set = self.create_data_set(data)
        train_loader = DataLoader(train_data_set, batch_size=1)

        print("Training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.train()
        for i in range(epochs):
            for texts, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch: {i+1}/{epochs} completed")

    def start_eval(self, data):
        self.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for d in data:
                texts, labels = d['text'], d['label']
                token_ids = self.tokenizer.encode(texts)
                outputs = self(torch.tensor(token_ids).unsqueeze(0).to(self.device))
                total += 1
                _, predicted = torch.max(outputs, 1)
                print(f"With test {texts} predicted: {predicted} actual: {labels}")
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {correct / total  * 100}%")

        