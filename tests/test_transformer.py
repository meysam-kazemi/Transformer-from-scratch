import torch

from src.transformer import Transformer


def test_transformer_forward_shape():
    src_vocab_size, tgt_vocab_size = 50, 60
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=64,
        num_heads=8,
        num_layers=2,
        d_ff=128,
        max_seq_length=20,
        dropout=0.1,
    )
    model.eval()

    batch_size, src_len, tgt_len = 2, 7, 5
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    out = model(src, tgt)
    assert out.shape == (batch_size, tgt_len, tgt_vocab_size)
