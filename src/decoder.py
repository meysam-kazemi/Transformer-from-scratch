import os
import sys
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.multi_head_attention import MultiHeadAttention
from src.feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
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
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
