"""
Train the char-GPT on Tiny Shakespeare, saving a checkpoint every
`checkpoint_every` steps. This dense checkpointing is the whole point
of this project: it's what lets the visualizer show HOW the model's
internals evolve over training, not just the final result.

Usage:
    python src/train.py                  # full run (~3000 steps, ~10-15 min CPU)
    python src/train.py --quick          # fast smoke test (~200 steps, ~1 min)
"""
import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPTConfig, GPT
from data import load_data, get_batch


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters=20):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--data_path", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--quick", action="store_true", help="fast smoke test: fewer steps, smaller model")
    args = parser.parse_args()

    if args.quick:
        args.steps = 200
        args.checkpoint_every = 20
        args.n_layer = 2
        args.n_embd = 64
        args.block_size = 64
        args.batch_size = 32

    os.makedirs(args.save_dir, exist_ok=True)

    train_data, val_data, tokenizer = load_data(args.data_path)
    print(f"Dataset: {len(train_data) + len(val_data):,} chars | vocab size: {tokenizer.vocab_size}")

    # Save tokenizer vocab so the visualizer can decode without retraining
    import json
    with open(os.path.join(args.save_dir, "tokenizer.json"), "w") as f:
        json.dump({"itos": tokenizer.itos, "stoi": tokenizer.stoi}, f)

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(args.device)
    print(f"Model params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = os.path.join(args.save_dir, "metrics.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "elapsed_sec"])

    start = time.time()
    for step in range(args.steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.checkpoint_every == 0 or step == args.steps:
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.device)
            elapsed = time.time() - start
            print(f"step {step:5d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | {elapsed:.0f}s")

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, loss.item(), val_loss, elapsed])

            ckpt_path = os.path.join(args.save_dir, f"step_{step:06d}.pt")
            torch.save({"model_state": model.state_dict(), "config": cfg, "step": step}, ckpt_path)

    print(f"Done. Checkpoints saved to {args.save_dir}/step_*.pt")
    print(f"Metrics log saved to {log_path}")


if __name__ == "__main__":
    main()
