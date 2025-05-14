import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=20, vocab_size=50):
        super().__init__()
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        # Generate random integer sequences as source and target (shift targets for teacher forcing).
        self.data = []
        for _ in range(num_samples):
            src = torch.randint(2, vocab_size, (seq_length,))
            tgt = torch.randint(2, vocab_size, (seq_length,))
            self.data.append((src, tgt))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
