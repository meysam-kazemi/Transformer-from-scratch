import torch
from torch import nn
from torch.functional import F

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism as described in the "Attention is All You Need" paper.

    This module divides the input embedding into multiple smaller heads, then computes attention independently 
    on each head before concatenating the results and projecting back to the original embedding dimension.

    Args:
        d_model (int): The dimensionality of the input embeddings. Default is 512.
        n_heads (int): Number of attention heads. Default is 8.

    Attributes:
        d (int): Dimensionality for each head computed as embed_dim // n_heads.
        query (nn.Linear): Linear projection layer for the query.
        key (nn.Linear): Linear projection layer for the key.
        value (nn.Linear): Linear projection layer for the value.
        fc (nn.Linear): Final fully connected layer to project concatenated outputs back to embed_dim.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.w_q(Q))
        K = self.split_heads(self.w_k(K))
        V = self.split_heads(self.w_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.w_o(self.combine_heads(attn_output))
        return output

