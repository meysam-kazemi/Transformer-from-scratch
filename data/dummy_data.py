import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenizer
from counter import Counter

dataset = load_dataset("opus-books", "en-nl", split="train")

SRC_LNG = "nl"
TGT_LNG = "en"

special_symbols = ["<unk>", "<pad>", "bos", "eos"]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def build_vocab(dataset, lang):
    counter = Counter()
    for item in dataset:
        counter.update(word_tokenizer(item['translation'][lang]))
    vocab = {sym:i for i, sym in enumerate(special_symbols)}
    for word in counter:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab



