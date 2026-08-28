"""Basic sanity tests. Run with: pytest tests/"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from model import GPTConfig, GPT
from data import CharTokenizer, get_batch


def test_tokenizer_roundtrip():
    text = "hello world"
    tok = CharTokenizer(text)
    encoded = tok.encode("hello")
    decoded = tok.decode(encoded)
    assert decoded == "hello"


def test_model_forward_shape():
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg)
    idx = torch.randint(0, 20, (4, 16))
    logits, loss = model(idx)
    assert logits.shape == (4, 16, 20)
    assert loss is None  # no targets passed


def test_model_forward_with_loss():
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg)
    idx = torch.randint(0, 20, (4, 16))
    targets = torch.randint(0, 20, (4, 16))
    logits, loss = model(idx, targets)
    assert loss.item() > 0


def test_causal_masking():
    """Attention pattern should be lower-triangular (no attending to future)."""
    cfg = GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg)
    idx = torch.randint(0, 20, (1, 8))
    model(idx)
    pattern = model.blocks[0].attn.last_attn_pattern[0, 0]  # [T, T]
    upper_triangle = torch.triu(pattern, diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-5)


def test_generate_produces_correct_length():
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg)
    idx = torch.randint(0, 20, (1, 4))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 14)


def test_get_batch_shapes():
    data = torch.arange(1000)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y should be x shifted by one
    assert torch.equal(y[0][:-1], x[0][1:])


def test_model_can_overfit_tiny_batch():
    cfg = GPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg)
    idx = torch.randint(0, 10, (4, 8))
    targets = torch.randint(0, 10, (4, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(50):
        _, loss = model(idx, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5
