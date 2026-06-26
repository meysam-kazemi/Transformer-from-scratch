# Transformer from Scratch

A from-scratch PyTorch implementation of the Transformer architecture from
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), trained for
English→Dutch translation on the `opus_books` dataset. Built to deepen
understanding of the core building blocks — multi-head attention, positional
encoding, encoder/decoder stacks — rather than to use a high-level library.

<img width="400" alt="Transformer architecture" src="https://github.com/user-attachments/assets/e84e0192-b134-4469-9c74-fb5b1a53dc63" />

## Project Structure

- `src/` — model components (attention, encoder, decoder, embeddings, etc.)
- `data/translation_data.py` — dataset loading, tokenization, and batching
- `train.py` — training entry point
- `tests/` — pytest unit tests

## Dataset

English→Dutch pairs from the [`opus_books`](https://huggingface.co/datasets/opus_books)
dataset, loaded via Hugging Face `datasets` and tokenized with NLTK.

## Setup

```bash
git clone https://github.com/meysam-kazemi/Transformer-from-scratch.git
cd Transformer-from-scratch
pip install -r requirements.txt
```

GPU is used automatically if a CUDA-compatible device is available.

## Usage

Run training from the repository root:

```bash
python train.py
```

## Reproducibility

A fixed seed (`SEED = 42` in `train.py`) makes runs reproducible. Edit the
constant to change it.

## Model Configuration

Defaults (see `train.py`): `d_model=512`, `num_heads=8`, `num_layers=6`,
`d_ff=2048`, `dropout=0.1`.

## Tests

```bash
pytest
```
