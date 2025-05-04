import numpy as np
import turch
from torch import nn
from torch.nn.functional import F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim:int=512, n_heads:int=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d = int(embed_dim/n_heads)

        # Query and Key and Value matrices
        self.query = nn.Linear(self.d, self.d, bias=False)
        self.key = nn.Linear(self.d, self.d, bias=False)
        self.value = nn.Linear(self.d, self.d, bias=False)




