from abc import abstractmethod
from torch import nn
import torch

class IModel(nn.Module): 
    @abstractmethod
    def start_train(self, data):
        pass

    @abstractmethod
    def start_eval(self, data):
        pass

    @abstractmethod
    def encode(self, data):
        pass

    @abstractmethod
    def decode(self, path):
        pass

    def set_device(self, device):
        self.to(device)
        self.device = device
        if device == "mps":
            self.to(torch.float32)