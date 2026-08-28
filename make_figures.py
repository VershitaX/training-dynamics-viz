"""
Renders static artifacts from the checkpoint history for the README:
  - figures/loss_curve.png
  - figures/embedding_evolution.png (small multiples: PCA at several steps)
  - figures/generation_samples.txt (text generated at several steps, side by side)

Run after training:
    python make_figures.py
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "src")
from analysis import list_checkpoints, load_tokenizer, load_checkpoint, generate_sample, embedding_pca_2d


def main():
    os.makedirs("figures", exist_ok=True)
    checkpoints = list_checkpoints("checkpoints")
    if not checkpoints:
        print("No checkpoints found. Run `python src/train.py` first.")
        return

    tokenizer = load_tokenizer("checkpoints")

    # --- Loss curve ---
    metrics_path = "checkpoints/metrics.csv"
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(df["step"], df["train_loss"], label="train loss", linewidth=2)
        ax.plot(df["step"], df["val_loss"], label="val loss", linewidth=2)
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.set_title("Training loss over time")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig("figures/loss_curve.png", dpi=150, bbox_inches="tight")
        print("saved figures/loss_curve.png")

    # --- Embedding evolution: small multiples across training ---
    n_show = min(6, len(checkpoints))
    idxs = [int(i * (len(checkpoints) - 1) / (n_show - 1)) for i in range(n_show)]
    fig, axes = plt.subplots(1, n_show, figsize=(3.2 * n_show, 3.2))
    if n_show == 1:
        axes = [axes]
    for ax, idx in zip(axes, idxs):
        step, path = checkpoints[idx]
        model = load_checkpoint(path)
        proj, chars = embedding_pca_2d(model, tokenizer)
        ax.scatter(proj[:, 0], proj[:, 1], s=15, alpha=0.7)
        ax.set_title(f"step {step}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Token embedding space (PCA) over training", y=1.05)
    fig.savefig("figures/embedding_evolution.png", dpi=150, bbox_inches="tight")
    print("saved figures/embedding_evolution.png")

    # --- Generation samples across training ---
    lines = []
    for step, path in checkpoints[:: max(1, len(checkpoints) // 6)]:
        model = load_checkpoint(path)
        sample = generate_sample(model, tokenizer, prompt="ROMEO:", max_new_tokens=150, temperature=0.8)
        lines.append(f"=== step {step} ===\n{sample}\n")
    with open("figures/generation_samples.txt", "w") as f:
        f.write("\n".join(lines))
    print("saved figures/generation_samples.txt")


if __name__ == "__main__":
    main()
