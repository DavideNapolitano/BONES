import numpy as np
import torch

class L1():
    def __init__(self):
        self.name = 'L1'

    def compute(self, y_true, y_pred):
        return torch.norm((y_true - y_pred), p=1)
    
class L2():
    def __init__(self):
        self.name = 'L2'

    def compute(self, y_true, y_pred):
        return torch.norm((y_true - y_pred), p=2)