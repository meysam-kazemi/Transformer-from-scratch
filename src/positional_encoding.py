import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in the "Attention is All You Need" paper.
    
    Positional encoding injects information about the position of tokens in the sequence.
    This implementation computes sinusoidal positional encodings which are added to the input embeddings.
    
    Args:
        d_model (int): Dimension of the embeddings.
        max_length (int): Maximum length of the sequences for which to generate positional encodings. Default is 5000.
        dropout (float): Dropout rate applied to the final output. Default is 0.1.
    
    Attributes:
        dropout (nn.Dropout): Dropout layer applied to the output.
        pe (torch.Tensor): Pre-computed positional encodings of shape (1, max_length, d_model) stored as a buffer.
    """
    def __init__(self, d_model: int, max_length: int=5000, dropout: float=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Initialize a matrix to hold positional encodings for max_length positions and d_model dimensions.
        positional_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )

        # For every even dimension, use sine function; for every odd dimension, use cosine function.
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        pe = positional_encoding.unsqueeze(0)

        # we use register_buffer to save the "pe" parameter to the state_dict
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        """
        Add positional encoding to the input tensor and apply dropout.
        
        Args:
            x (torch.Tensor): Input embeddings tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: The input tensor with positional encodings added and dropout applied.
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



