import os
import sys
from torch import nn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.multi_head_attention import MultiHeadAttention
from src.feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
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
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
