import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from collections import Counter

dataset = load_dataset("opus_books", "en-nl", split="train")

SRC_LNG = "en"
TGT_LNG = "nl"

special_symbols = ["<unk>", "<pad>", "bos", "eos"]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def build_vocab(dataset, lang):
    counter = Counter()
    for item in dataset:
        counter.update(word_tokenize(item['translation'][lang]))
    vocab = {sym:i for i, sym in enumerate(special_symbols)}
    for word in counter:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

import pdb; pdb.set_trace()
src_vocab = build_vocab(dataset, SRC_LNG)
tgt_vocab = build_vocab(dataset, TGT_LNG)

print(type(src_vocab))


