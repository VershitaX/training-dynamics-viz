"""
Character-level tokenizer and batching for the Tiny Shakespeare dataset.
Deliberately simple (no BPE) so the vocabulary is small and every token
is a single human-readable character -- makes the embedding-space
visualizations directly interpretable (you can label each point with
the literal character it represents).
"""
import torch


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def load_data(path: str = "data/tinyshakespeare.txt", train_frac: float = 0.9):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(data) * train_frac)
    return data[:n], data[n:], tokenizer


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device="cpu"):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
