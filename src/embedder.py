import numpy as np
from torch import nn

class Embedder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedder = nn.Embedder(vocab_size, d_model)

    def forward(self, x):
        x = self.embedder(x) * np.sqrt(self.d_model)
        return x




