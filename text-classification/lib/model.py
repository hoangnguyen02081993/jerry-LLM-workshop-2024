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
    
    def start_train(self, data, test_data, epochs=10):
        train_data_set = self.create_data_set(data)
        train_loader = DataLoader(train_data_set, batch_size=2)

        print("Training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0004, weight_decay=0.1)

        self.train()
        for _ in range(epochs):
            for texts, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # self.eval()
        # with torch.no_grad():
        #     train_loss = self.calc_loss_loader(train_loader, self, self.device)
        #     print("Train loss: ", train_loss)

        