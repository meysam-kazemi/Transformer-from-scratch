import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from collections import Counter

SRC_LNG = "en"
TGT_LNG = "nl"
BATCH_SIZE = "8"

train_data = load_dataset("opus_books", "en-nl", split="train[:90%]")
valid_data = load_dataset("opus_books", "en-nl", split="train[90%:]")

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

# Ta inja okeye

def tokenize(text):
    return word_tokenize(text.lower())

def encode(text, vocab):
    return [vocab.get(tok, UNK_IDX) for tok in tokenize(text)]

def numerify(item, vocab, lang):
    encoded = encode(item['translation'][lang], vocab)
    return torch.tensor([BOS_IDX]+encoded+[EOS_IDX], dtype=torch.long)

def prepare_batch(batch):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(numerify(item, src_vocab, SRC_LNG))
        tgt_batch.append(numerify(item, tgt_vocab, TGT_LNG))

        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=prepare_batch)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=prepare_batch)

        




