import numpy as np
from torch import nn

class Embedder(nn.Module):
    """
    A simple embedding layer that converts token indices into dense embeddings
    and scales the embeddings by the square root of the model dimension.

    This scaling (multiplying by sqrt(d_model)) is used in the Transformer architecture
    to counteract the variance introduced by the embedding layer.
    
    Args:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        d_model (int): The dimensionality of each embedding vector.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedder = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Convert token indices to embeddings and scale the embeddings.

        Args:
            x (torch.Tensor): Input tensor containing token indices 
                              with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Scaled embeddings with shape (batch_size, sequence_length, d_model).
        """
        x = self.embedder(x) * np.sqrt(self.d_model)
        return x

if __name__=="__main__":
    embedder = Embedder(3000, 768)


