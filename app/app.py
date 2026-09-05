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
    label = "benign" if p_benign >= 0.5 else "malignant"
    verdict = (
        f"**Model output:** `{p_benign:.4f}` on a 0-1 scale where 1 is benign and "
        f"0 is malignant, so at the 0.50 threshold it reads this sample as "
        f"**{label}**.\n\n"
        f"Trained on 455 cases; this one was held out."
    )
    return verdict, {"Benign": p_benign, "Malignant": 1 - p_benign}


def example_case(index):
    """Pull a real case out of the dataset so a visitor can try it in one click."""
    i = int(index) % len(DATA.data)
    actual = "benign" if DATA.target[i] == 1 else "malignant"
    return ", ".join(f"{v:g}" for v in DATA.data[i]), f"### Case {i} — recorded diagnosis: {actual}"


with gr.Blocks(title="Neural network from scratch — breast cancer classifier") as demo:
    gr.Markdown(
        "# Neural network from scratch\n"
        "### A 3-layer classifier written by hand in NumPy — no PyTorch, no TensorFlow, "
        "no autograd\n\n"
        "> **This is a learning project, not a medical device.** It is trained on 455 "
        "cases from a single 1990s research dataset, has no clinical validation of any "
        "kind, and must not be used to make any decision about anyone's health.\n\n"
        "Pick any of the 569 research cases below and compare what the network scores it "
        "against the diagnosis recorded in the dataset. The interesting ones are the "
        "three it gets wrong: **73**, **541** and **542**."
    )

    with gr.Row():
        index = gr.Slider(0, len(DATA.data) - 1, value=73, step=1,
                          label="Research case number")
        run = gr.Button("Run the model", variant="primary")

    actual = gr.Markdown()
    verdict = gr.Markdown()
    scores = gr.Label(label="Model score", num_top_classes=2)

    with gr.Accordion("The 30 measurements for this case", open=False):
        raw = gr.Textbox(label="", lines=4, interactive=False)

    def show(i):
        measurements, actual_md = example_case(i)
        v, sc = diagnose(measurements)
        return measurements, actual_md, v, sc

    run.click(show, inputs=index, outputs=[raw, actual, verdict, scores])
    demo.load(show, inputs=index, outputs=[raw, actual, verdict, scores])

    gr.Markdown(
        "The 30 features are: " + ", ".join(FEATURES) + ".\n\n"
        "Source: Breast Cancer Wisconsin (Diagnostic), UCI dataset 17."
    )

if __name__ == "__main__":
    demo.launch()
