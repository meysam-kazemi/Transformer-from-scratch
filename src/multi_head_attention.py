import numpy as np
import torch
from torch import nn
from torch.functional import F


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

        # fully connected layer.
        self.fc = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        key_len, query_len, value_len = key.size(1), query.size(1), value.size(1)

        # reshape from (batch_size x seq_len x embed_size) -> (batch_size x seq_len x heads x head)
        # example: from (32x10x512) -> (32x10x8x64)
        key = key.reshape(batch_size, key_len, self.heads, self.head)
        query = query.reshape(batch_size, query_len, self.n_heads, self.d)
        value = value.reshape(batch_size, value_len, self.n_heads, self.d)

        key = self.key(key)  # (32x10x8x64)
        query = self.query(query)  # (32x10x8x64)
        value = self.value(value)  # (32x10x8x64)


        ############### query x key ###############

        # query shape: batch_size x q_len, heads, head, e.g: (32x10x8x64)
        # key shape: batch_size x v_len, heads, head, e.g: (32x10x8x64)
        # product shape should be: batch_size, heads, q_len, v_len, e.g: (32x8x10x10)
        product = torch.einsum("bqhd,bkhd->bhqk", [query, key])

        # if mask (in decoder)
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / sqrt(self.head)

        scores = F.softmax(product, dim=-1)

        ############### scores x value ###############

        # scores shape: batch_size, heads, q_len, v_len, e.g: (32x8x10x10)
        # value shape: batch_size, v_len, heads, head, e.g: (32x10x8x64)
        # output: batch_size, heads, v_len, head, e.g: (32x10x512)

        output = torch.einsum("nhql,nlhd->nqhd", [scores, value]).reshape(
            batch_size, query_len, self.heads * self.d
        )

        output = self.fc(output)  # (32x10x512) -> (32x10x512)

        return output




multi_head_attention = MultiHeadAttention()

