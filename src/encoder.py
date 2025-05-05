import os
import sys
import numpy as np
import torch
from torch import nn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.embedder import Embedder
from src.multi_head_attention import MultiHeadAttention
from src.positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(
        self,
        seq_len,
        vocab_size,
        embed_dim,
        num_blocks=6,
        expantion_factor=4,
        heads=8,
        dropout=0.2,
    ):
        """
        The Encoder part of the Transformer architecture
        it is a set of stacked encoders on top of each others, in the paper they used stack of 6 encoders

        :param seq_len: the length of the sequence, in other words, the length of the words
        :param vocab_size: the total size of the vocabulary
        :param embed_dim: the embedding dimension
        :param num_blocks: the number of blocks (encoders), 6 by default
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer in each encoder
        :param heads: the number of heads in each encoder
        :param dropout: probability dropout (between 0 and 1)
        """
        super().__init__()
        self.embedding = Embedder(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, seq_len)
        

