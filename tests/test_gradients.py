"""Check the hand-derived backward pass against PyTorch autograd.

The whole claim of this project is that the gradients were worked out by hand rather
than handed to an autograd engine. That claim needs a test, because a wrong gradient
does not announce itself: it still points roughly downhill, the loss still falls, and
the accuracy can still look fine. This suite is what turns "derived by hand" into
"derived by hand and verified".

It caught a real bug. The output layer applied the sigmoid derivative on top of
dA = A - y, which is the gradient for mean squared error, not for binary cross-entropy.
Sigmoid paired with BCE cancels to dZ = A - y exactly. The spurious a(1-a) factor
averaged about 0.23, so the output layer trained at roughly a quarter of its intended
learning rate -- which is why the network used to look underfitted after 300 epochs.

PyTorch is used ONLY as an oracle here. It never trains the model and is not a runtime
dependency; the shipped network is pure NumPy.

    pip install -r requirements-dev.txt
    pytest tests/ -v
"""
import json
import pathlib

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "NeuralNet_BreastCancer.ipynb"
LAYERS = [4, 5, 3, 1]
SEED = 1


def load_from_notebook():
    """Execute the notebook's own definition cells.

    Importing a copy would test the copy. The notebook is the artifact this project
    ships, so the test runs the definitions straight out of it.
    """
    nb = json.loads(NOTEBOOK.read_text())
    ns = {"np": np}
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "def relu" in src or "class NeuralNN" in src:
            exec(compile(src, str(NOTEBOOK), "exec"), ns)
    assert "NeuralNN" in ns, "NeuralNN not found in the notebook"
    return ns


@pytest.fixture(scope="module")
def ns():
    return load_from_notebook()


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, LAYERS[0]))
    y = (rng.random((7, 1)) > 0.5).astype(float)
    return X, y


def hand_gradients(NeuralNN, X, y):
    """Recover dW and db from the in-place update, using lr = 1 and no weight decay.

    Returns the weights the gradients were taken AT -- `_backward` mutates the network,
    so handing the updated net to the oracle would differentiate at the wrong point.
    """
    net = NeuralNN(layers=LAYERS, lr=1.0, l2=0.0, seed=SEED)
    W0 = [w.copy() for w in net.W]
    b0 = [b.copy() for b in net.b]
    A, Z = net._forward(X)
    net._backward(A, Z, y)
    dW = [W0[i] - net.W[i] for i in range(len(W0))]
    db = [b0[i] - net.b[i] for i in range(len(b0))]
    return (W0, b0), dW, db


def torch_gradients(weights, X, y):
    """The same architecture and weights in torch.nn, differentiated by autograd."""
    W_at, b_at = weights
    Ws = [torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in W_at]
    bs = [torch.tensor(b, dtype=torch.float64, requires_grad=True) for b in b_at]
    a = torch.tensor(X, dtype=torch.float64)
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = a @ W + b
        a = torch.sigmoid(z) if i == len(Ws) - 1 else torch.relu(z)
    loss = torch.nn.functional.binary_cross_entropy(
        a, torch.tensor(y, dtype=torch.float64))
    value = float(loss.detach())
    loss.backward()
    return [W.grad.numpy() for W in Ws], [b.grad.numpy() for b in bs], value


def test_weight_gradients_match_autograd(ns, batch):
    X, y = batch
    at, dW, _ = hand_gradients(ns["NeuralNN"], X, y)
    tW, _, _ = torch_gradients(at, X, y)
    for i, (mine, theirs) in enumerate(zip(dW, tW)):
        np.testing.assert_allclose(
            mine, theirs, rtol=1e-9, atol=1e-10,
            err_msg=f"layer {i} weight gradient disagrees with autograd")


def test_bias_gradients_match_autograd(ns, batch):
    X, y = batch
    at, _, db = hand_gradients(ns["NeuralNN"], X, y)
    _, tb, _ = torch_gradients(at, X, y)
    for i, (mine, theirs) in enumerate(zip(db, tb)):
        np.testing.assert_allclose(
            mine.ravel(), theirs.ravel(), rtol=1e-9, atol=1e-10,
            err_msg=f"layer {i} bias gradient disagrees with autograd")


def test_loss_matches_torch(ns, batch):
    """The hand-written BCE agrees with torch's, so the gradients above are of the
    same objective rather than of two different losses that happen to look similar."""
    X, y = batch
    net = ns["NeuralNN"](layers=LAYERS, lr=1.0, l2=0.0, seed=SEED)
    mine = ns["bce"](y, net._forward(X)[0][-1])
    _, _, theirs = torch_gradients(([w.copy() for w in net.W], [b.copy() for b in net.b]), X, y)
    # bce() adds eps = 1e-8 inside the logs to stay finite at 0 and 1, so the two
    # values agree to that epsilon rather than exactly.
    assert abs(mine - theirs) < 1e-7, f"BCE {mine} vs torch {theirs}"


def test_gradients_match_finite_differences(ns, batch):
    """A second opinion that needs no autograd at all, in case torch is unavailable
    or wrong about something. Slower, so it runs on the smallest layer only."""
    X, y = batch
    NeuralNN, bce = ns["NeuralNN"], ns["bce"]
    _, dW, _ = hand_gradients(NeuralNN, X, y)

    probe = NeuralNN(layers=LAYERS, lr=1.0, l2=0.0, seed=SEED)
    eps = 1e-6
    li = len(probe.W) - 1
    it = np.nditer(probe.W[li], flags=["multi_index"])
    while not it.finished:
        ix = it.multi_index
        original = probe.W[li][ix]
        probe.W[li][ix] = original + eps
        up = bce(y, probe._forward(X)[0][-1])
        probe.W[li][ix] = original - eps
        down = bce(y, probe._forward(X)[0][-1])
        probe.W[li][ix] = original
        assert abs(dW[li][ix] - (up - down) / (2 * eps)) < 1e-6, (
            f"output-layer weight {ix} disagrees with finite differences")
        it.iternext()


def test_sigmoid_bce_collapse_is_implemented(ns, batch):
    """The specific bug this suite was written for.

    For the output layer, dZ must equal A - y. If the sigmoid derivative is applied
    again, every output gradient is scaled by a(1-a) and this ratio is not 1.
    """
    X, y = batch
    at, dW, _ = hand_gradients(ns["NeuralNN"], X, y)
    tW, _, _ = torch_gradients(at, X, y)
    last = len(dW) - 1
    nonzero = np.abs(tW[last]) > 1e-12
    ratio = dW[last][nonzero] / tW[last][nonzero]
    assert np.allclose(ratio, 1.0, atol=1e-8), (
        f"output-layer gradients are scaled by {ratio.mean():.4f} rather than 1.0 — "
        "the sigmoid derivative is being applied on top of dA = A - y")
