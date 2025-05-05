import os
import sys
import numpy as np
import torch
from torch import nn
sys.path.append(os.path.join(os.path.dirname(__file__, "..")))
from src.multi_head_attention import MultiHeadAttention
from src.embedder import Embedder
from src.positional_encoding import PositionalEncoding


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim=512,
                 heads=8,
                 expansion_factor=4,
                 dropout=0.2
                 ):
        """
        The Transformer Block for the Encode and Decoder

        :param embed_dim: the embedding dimension
        :param heads: the number of heads
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        """
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, heads)  # the multi-head attention
        self.norm = nn.LayerNorm(embed_dim)  # the normalization layer

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),  # e.g: 512x(4*512) -> (512, 2048)
            nn.ReLU(),  # ReLU activation function
            nn.Linear(embed_dim * expansion_factor, embed_dim),  # e.g: 4*512)x512 -> (2048, 512)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, value, mask=None):
        # first, pass the key, query and value through the multi head attention layer
        attention_out = self.attention(query, key, value, mask)  # e.g.: 32x10x512

        # then add the residual connection
        attention_out = attention_out + value  # e.g.: 32x10x512

        # after that we normalize and use dropout
        attention_norm = self.dropout(self.norm(attention_out))  # e.g.: 32x10x512
        # print(attention_norm.shape)

        fc_out = self.feed_forward(attention_norm)  # e.g.:32x10x512 -> #32x10x2048 -> 32x10x512

        # Residual connection
        fc_out = fc_out + attention_norm  # e.g.: 32x10x512

        # dropout + normalization
        fc_norm = self.dropout(self.norm(fc_out))  # e.g.: 32x10x512

        return fc_norm
