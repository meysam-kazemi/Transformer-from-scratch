import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int=5000, dropout: float=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        positional_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(log(10000.0) / d_model)
        )

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        pe = positional_encoding.unsqueeze(0)

        # we use register_buffer to save the "pe" parameter to the state_dict
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
