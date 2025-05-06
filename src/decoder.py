import os
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import replicate
from src.multi_head_attentio import MultiHeadAttention
from src.positional_encoding import PositionalEncoding
from src.blocks import MuliDecoderBlock


class Decoder(nn.Module):

    def __init__(self,
                 target_vocab_size,
                 seq_len,
                 embed_dim=512,
                 num_blocks=6,
                 expansion_factor=4,
                 heads=8,
                 dropout=0.2
                 ):
        """
        The Decoder part of the Transformer architecture

    it is a set of stacked decoders on top of each others, in the paper they used stack of 6 decoder
        :param target_vocab_size: the size of the target
        :param seq_len: the length of the sequence, in other words, the length of the words
        :param embed_dim: the embedding dimension
        :param num_blocks: the number of blocks (encoders), 6 by default
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer in each decoder
        :param heads: he number of heads in each decoder
        :param dropout: probability dropout (between 0 and 1)
        """
        super().__init__()

        # define the embedding
        self.embedding = nn.Embedding(target_vocab_size, embed_dim)
        # the positional embedding
        self.positional_encoder = PositionalEncoding(embed_dim, seq_len)

        # define the set of decoders
        self.blocks = MuliDecoderBlock(
            num_blocks=num_blocks,
            embed_idm=embed_dim,
            heads=heads,
            expension_factor=expansion_factor,
            dropout=dropout
            ).blocks
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, encoder_output, mask):
        x = self.dropout(self.positional_encoder(self.embedding(x)))  # 32x10x512

        for block in self.blocks:
            x = block(encoder_output, x, encoder_output, mask)

        return x
