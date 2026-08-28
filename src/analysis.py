"""
Core analysis utilities for exploring how the model changes across
training checkpoints. Everything here is checkpoint-indexed so the
Streamlit app can scrub through training steps with a slider.
"""
import glob
import json
import os
import re

import numpy as np
import torch

from model import GPTConfig, GPT
from data import CharTokenizer


def list_checkpoints(save_dir: str = "checkpoints"):
    """Returns sorted list of (step, filepath) for all saved checkpoints."""
    paths = glob.glob(os.path.join(save_dir, "step_*.pt"))
    steps_paths = []
    for p in paths:
        m = re.search(r"step_(\d+)\.pt", p)
        if m:
            steps_paths.append((int(m.group(1)), p))
    return sorted(steps_paths, key=lambda x: x[0])


def load_tokenizer(save_dir: str = "checkpoints"):
    with open(os.path.join(save_dir, "tokenizer.json")) as f:
        d = json.load(f)
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.itos = {int(k): v for k, v in d["itos"].items()}
    tok.stoi = d["stoi"]
    tok.vocab_size = len(tok.itos)
    return tok


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def generate_sample(model, tokenizer, prompt: str = "\n", max_new_tokens: int = 200, temperature: float = 0.8, seed: int = 0):
    torch.manual_seed(seed)
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=20)
    return tokenizer.decode(out[0].tolist())


def embedding_pca_2d(model, tokenizer, n_components: int = 2):
    """PCA-projects the token embedding matrix down to 2D so it can be
    scatter-plotted, with each point labeled by its character. Uses
    plain numpy SVD (no sklearn dependency needed)."""
    W = model.tok_emb.weight.detach().numpy()  # [vocab_size, n_embd]
    W_centered = W - W.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
    proj = W_centered @ Vt[:n_components].T  # [vocab_size, n_components]
    chars = [tokenizer.itos[i] for i in range(tokenizer.vocab_size)]
    return proj, chars


def attention_snapshot(model, tokenizer, prompt: str, layer: int = 0):
    """Runs the model on `prompt` and returns the attention pattern
    (averaged over heads) for the given layer, plus the character
    labels for each position."""
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    pattern = model.attention_snapshot(idx, layer=layer)  # [1, nh, T, T]
    pattern = pattern[0].mean(dim=0).numpy()  # [T, T]
    chars = list(prompt)
    return pattern, chars
