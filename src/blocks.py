import os
import sys
import copy
from torch import nn
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.multi_head_attention import MultiHeadAttention


class EncoderBlock(nn.Module):
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
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, heads)  # the multi-head attention
        self.norm = nn.LayerNorm(embed_dim)  # the normalization layer

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),  # e.g: 512x(4*512) -> (512, 2048)
            nn.ReLU(),  # ReLU activation function
            nn.Linear(embed_dim * expansion_factor, embed_dim),  # e.g: 4*512)x512 -> (2048, 512)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
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


class MultiEncoderBlock(nn.Module):
    def __init__(
        self,
        num_blocks=6,
        embed_dim=512,
        heads=8,
        expansion_factor=4,
        dropout=0.2
        ):
        self.blocks = nn.ModuleList([copy.deepcopy(Block(embed_dim, heads, expansion_factor, dropout)) for _ in range(num_blocks)])
        


class DecoderBlock(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 heads=8,
                 expansion_factor=4,
                 dropout=0.2
                 ):
        """
        The DecoderBlock which will consist of the TransformerBlock used in the encoder, plus a decoder multi-head attention
        :param embed_dim: the embedding dimension
        :param heads: the number of heads
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        """
        super(DecoderBlock, self).__init__()

        # First define the Decoder Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, heads)
        # normalization
        self.norm = nn.LayerNorm(embed_dim)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)
        # finally th transformerBlock
        self.encoder_block = EncoderBlock(embed_dim, heads, expansion_factor, dropout)

    def forward(self, query, key, x, mask):
        # pass the inputs to the decoder multi-head attention
        decoder_attention = self.attention(x, x, x, mask)
        # residual connection + normalization
        value = self.dropout(self.norm(decoder_attention + x))
        # finally the transformerBlock (multi-head attention -> residual + norm -> feed forward -> residual + norm)
        decoder_attention_output = self.encoder_block(query, key, value)

        return decoder_attention_output

class MultiDecoderBlock(nn.Module):
    def __init__(
        self,
        num_blocks=6,
        embed_dim=512,
        heads=8,
        expansion_factor=4,
        dropout=0.2
        ):
        self.blocks = nn.ModuleList([copy.deepcopy(DecoderBlock(embed_dim, heads, expansion_factor, dropout)) for _ in range(num_blocks)])
