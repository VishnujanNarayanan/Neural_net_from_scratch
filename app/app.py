"""Gradio demo: diagnose a breast tumour from its 30 nucleus measurements.

Inference only. The network is trained in NeuralNet_BreastCancer.ipynb, which
exports the weights and the scaler statistics to model/weights.npz; this app
just runs the forward pass, so the Space needs no training step and starts cold
in a second.
"""
from pathlib import Path

import gradio as gr
import numpy as np
from sklearn.datasets import load_breast_cancer

HERE = Path(__file__).resolve().parent
# On Hugging Face the Space root IS this directory, so model/ sits beside app.py.
# In the git repo it sits one level up. Take whichever exists.
_CANDIDATES = [HERE / "model" / "weights.npz", HERE.parent / "model" / "weights.npz"]
WEIGHTS = next((c for c in _CANDIDATES if c.exists()), None)
if WEIGHTS is None:
    raise FileNotFoundError(
        "model/weights.npz not found. Run the notebook to export it, then copy "
        "model/ into this directory before deploying the Space."
    )

ART = np.load(WEIGHTS, allow_pickle=False)

W = [ART[f"W{i}"] for i in range(int(ART["n_layers"]))]
B = [ART[f"b{i}"] for i in range(int(ART["n_layers"]))]
MEAN, SCALE = ART["scaler_mean"], ART["scaler_scale"]

DATA = load_breast_cancer()
FEATURES = list(DATA.feature_names)


def forward(x):
    """Same maths as NeuralNN._forward: ReLU hidden layers, sigmoid output."""
    a = (np.asarray(x, dtype=float).reshape(1, -1) - MEAN) / SCALE
    for i, (w, b) in enumerate(zip(W, B)):
        z = a @ w + b
        a = 1 / (1 + np.exp(-z)) if i == len(W) - 1 else np.maximum(0, z)
    return float(a.ravel()[0])


def diagnose(raw_csv):
    values = [v for v in raw_csv.replace("\n", ",").split(",") if v.strip()]
    if len(values) != 30:
        return f"Expected 30 measurements, got {len(values)}.", None
    try:
        x = [float(v) for v in values]
    except ValueError:
        return "Every measurement must be a number.", None

    p_benign = forward(x)
    label = "Benign" if p_benign >= 0.5 else "Malignant"
    verdict = (
        f"### {label}\n\n"
        f"The network scores this sample **{p_benign:.3f}** on a 0-1 scale where 1 is "
        f"benign and 0 is malignant. It was trained on 455 cases and has never seen "
        f"this one.\n\n"
        f"_A screening aid built as a learning project — not a medical device, and "
        f"not a substitute for a pathologist._"
    )
    return verdict, {"Benign": p_benign, "Malignant": 1 - p_benign}


def example_case(index):
    """Pull a real case out of the dataset so a visitor can try it in one click."""
    i = int(index) % len(DATA.data)
    actual = "benign" if DATA.target[i] == 1 else "malignant"
    return ", ".join(f"{v:g}" for v in DATA.data[i]), f"Case {i} — actual diagnosis: **{actual}**"


with gr.Blocks(title="Breast tumour diagnosis — neural network from scratch") as demo:
    gr.Markdown(
        "# Breast tumour diagnosis\n"
        "A 3-layer neural network written by hand in NumPy — no PyTorch, no TensorFlow, "
        "no autograd. It reads 30 measurements taken from a cell-nucleus image and says "
        "whether the growth looks malignant or benign.\n\n"
        "Load an example case below, or paste your own 30 comma-separated measurements."
    )

    with gr.Row():
        index = gr.Number(value=0, precision=0, label="Example case number (0-568)")
        load = gr.Button("Load example case")
    actual = gr.Markdown()

    raw = gr.Textbox(
        label="30 measurements, comma separated",
        lines=4,
        value=", ".join(f"{v:g}" for v in DATA.data[0]),
    )
    run = gr.Button("Diagnose", variant="primary")

    verdict = gr.Markdown()
    scores = gr.Label(label="Model confidence", num_top_classes=2)

    load.click(example_case, inputs=index, outputs=[raw, actual])
    run.click(diagnose, inputs=raw, outputs=[verdict, scores])

    gr.Markdown(
        "The 30 features are: " + ", ".join(FEATURES) + ".\n\n"
        "Source: Breast Cancer Wisconsin (Diagnostic), UCI dataset 17."
    )

if __name__ == "__main__":
    demo.launch()
