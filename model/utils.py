import numpy as np

class LogTransform:
    def __init__(self, bias, min_x=0):
        self.bias, self.min_x = bias, min_x

    def trf(self, x):
        return np.log2(x + self.bias) - np.log2(self.min_x + self.bias)

    def inv(self, x):
        return (2 ** (x + np.log2(self.min_x + self.bias))) - self.bias