"""Export the trained network and the dataset to JSON for the static browser demo.

The network is 641 parameters and three matrix multiplies, so it does not need a
server: docs/index.html runs the identical forward pass in JavaScript. This script
is the single source of truth for what that page loads, and CI reruns it so the
page can never drift from the notebook's weights.
"""
import json
import pathlib

import numpy as np
from sklearn.datasets import load_breast_cancer

ROOT = pathlib.Path(__file__).resolve().parents[1]
art = np.load(ROOT / "model" / "weights.npz", allow_pickle=False)
n = int(art["n_layers"])

model = {
    "W": [art[f"W{i}"].tolist() for i in range(n)],
    "b": [art[f"b{i}"].ravel().tolist() for i in range(n)],
    "scaler_mean": art["scaler_mean"].tolist(),
    "scaler_scale": art["scaler_scale"].tolist(),
}

data = load_breast_cancer()
cases = {
    "feature_names": list(data.feature_names),
    # rounded to 6 significant figures: identical predictions, much smaller payload
    "X": [[float(f"{v:.6g}") for v in row] for row in data.data],
    "y": [int(t) for t in data.target],  # 1 = benign, 0 = malignant
}

out = ROOT / "docs"
out.mkdir(exist_ok=True)
(out / "model.json").write_text(json.dumps(model, separators=(",", ":")))
(out / "cases.json").write_text(json.dumps(cases, separators=(",", ":")))

params = sum(np.asarray(w).size for w in model["W"]) + sum(len(b) for b in model["b"])
print(f"docs/model.json  {params} parameters, {(out / 'model.json').stat().st_size:,} bytes")
print(f"docs/cases.json  {len(cases['X'])} cases, {(out / 'cases.json').stat().st_size:,} bytes")
