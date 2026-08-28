"""
Training Dynamics Visualizer — scrub through a language model's training
and watch its internals evolve.

Run with:
    streamlit run app.py

Requires checkpoints/ to be populated by src/train.py first (run
`python src/train.py` for the full run, or `python src/train.py --quick`
for a fast smoke test).
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from analysis import (
    list_checkpoints,
    load_tokenizer,
    load_checkpoint,
    generate_sample,
    embedding_pca_2d,
    attention_snapshot,
)

st.set_page_config(page_title="Training Dynamics Visualizer", layout="wide")

st.title("Training Dynamics Visualizer")
st.caption("Watch a character-level GPT learn Shakespeare, checkpoint by checkpoint.")

CKPT_DIR = "checkpoints"
checkpoints = list_checkpoints(CKPT_DIR)

if not checkpoints:
    st.error(
        "No checkpoints found in `checkpoints/`. Run `python src/train.py --quick` "
        "(fast, ~1 min) or `python src/train.py` (full run, ~20 min) first, then reload this app."
    )
    st.stop()

tokenizer = load_tokenizer(CKPT_DIR)
steps = [s for s, _ in checkpoints]


@st.cache_resource(show_spinner=False)
def _load(step_to_path_str):
    """Cache keyed on the step (streamlit cache needs hashable args)."""
    path = dict(checkpoints)[int(step_to_path_str)]
    return load_checkpoint(path)


step = st.select_slider(
    "Training step — drag to scrub through training",
    options=steps,
    value=steps[-1],
)
model = _load(str(step))

col_left, col_right = st.columns([1, 1])

# ---------------------------------------------------------------
with col_left:
    st.subheader("What the model generates at this point")
    prompt = st.text_input("Prompt", value="ROMEO:")
    sample = generate_sample(model, tokenizer, prompt=prompt, max_new_tokens=200, temperature=0.8)
    st.code(sample, language=None)

    st.subheader("Loss curve")
    metrics_path = os.path.join(CKPT_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        st.line_chart(df.set_index("step")[["train_loss", "val_loss"]])
        current_loss = df[df["step"] == step]
        if len(current_loss):
            st.caption(
                f"At step {step}: train_loss={current_loss['train_loss'].values[0]:.3f}, "
                f"val_loss={current_loss['val_loss'].values[0]:.3f}"
            )

# ---------------------------------------------------------------
with col_right:
    st.subheader("Token embedding space (PCA, 2D)")
    st.caption("Each point is one character. Watch clusters form as training progresses — e.g. vowels grouping, punctuation separating from letters.")
    proj, chars = embedding_pca_2d(model, tokenizer)
    chart_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "char": [repr(c)[1:-1] or "·" for c in chars]})
    st.scatter_chart(chart_df, x="x", y="y", size=40)
    with st.expander("Show character labels"):
        st.dataframe(chart_df, hide_index=True)

    st.subheader("Attention pattern")
    attn_prompt = st.text_input("Text to inspect attention on", value="ROMEO:", key="attn_prompt")
    layer_choice = st.slider("Layer", 0, model.cfg.n_layer - 1, 0)
    pattern, attn_chars = attention_snapshot(model, tokenizer, attn_prompt, layer=layer_choice)
    # Position-tag labels (e.g. "O_1", "O_4") so repeated characters in the
    # prompt don't collide as duplicate row/column names -- pandas' Styler
    # errors on duplicate labels, which is exactly what "ROMEO:" triggers
    # (the letter O appears twice) without this.
    unique_labels = [f"{c}_{i}" for i, c in enumerate(attn_chars)]
    attn_df = pd.DataFrame(pattern, index=unique_labels, columns=unique_labels)
    st.dataframe(attn_df.style.background_gradient(cmap="viridis", axis=None).format("{:.2f}"))
    st.caption("Rows = query position (which character is 'looking'), columns = key position (which character it attends to).")

st.divider()
st.caption(
    f"Model: {model.cfg.n_layer} layers, {model.cfg.n_embd} dim, "
    f"{sum(p.numel() for p in model.parameters()):,} params. "
    f"Checkpoint at step {step} of {steps[-1]}."
)
