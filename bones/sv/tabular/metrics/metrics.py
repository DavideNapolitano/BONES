import numpy as np
from scipy.stats import kendalltau

class L1():
    def __init__(self):
        self.name = 'L1'

    def compute(self, y_true, y_pred):
        return np.linalg.norm(y_true-y_pred, ord=1)
    
class L2():
    def __init__(self):
        self.name = 'L2'

    def compute(self, y_true, y_pred):
        return np.linalg.norm(y_true-y_pred, ord=2)
    
class Kendall():
    def __init__(self):
        self.name = 'Kendall'

    def compute(self, y_true, y_pred):
        return kendalltau(y_true, y_pred)[0]