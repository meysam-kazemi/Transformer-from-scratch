import torch

from src.multi_head_attention import MultiHeadAttention


def test_mha_output_shape():
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    x = torch.randn(2, 10, 64)
    out = mha(x, x, x)
    assert out.shape == (2, 10, 64)


def test_mha_requires_divisible_dims():
    try:
        MultiHeadAttention(d_model=65, num_heads=8)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for indivisible d_model/num_heads")
