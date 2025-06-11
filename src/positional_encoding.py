import numpy as np
import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in the "Attention is All You Need" paper.
    
    Positional encoding injects information about the position of tokens in the sequence.
    This implementation computes sinusoidal positional encodings which are added to the input embeddings.
    
    Args:
        d_model (int): Dimension of the embeddings.
        max_length (int): Maximum length of the sequences for which to generate positional encodings. Default is 5000.
    
    Attributes:
        pe (torch.Tensor): Pre-computed positional encodings of shape (1, max_length, d_model) stored as a buffer.
    """
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

