# Neural Network Classifier – NumPy Implementation

This repository provides a clean, fully vectorized feed‑forward neural‑network classifier written from scratch in Python with NumPy. The implementation is self‑contained in `neural_nn_class.py` and is suitable for binary‑classification tasks such as the Breast‑Cancer‑Wisconsin diagnostic dataset.

---

## Repository Contents

```
Neural_networks/
├── neural_nn_class.py   # Core NeuralNN class (forward, backprop, training loop)
├── requirements.txt     # Minimal dependency list
└── README.md            # Project documentation
```

---

## Key Features

* Multi‑layer architecture defined by a simple list (e.g. `[30, 16, 8, 1]`)
* ReLU activation for hidden layers; Sigmoid activation for output
* Binary cross‑entropy loss
* Mini‑batch gradient descent with shuffle at every epoch
* L2 weight‑decay regularisation
* Utility methods for probability prediction and hard‑label prediction
* Designed for clarity and extensibility—suitable as a learning reference or a lightweight baseline

---

## Installation

```bash
# Clone repository
git clone https://github.com/<your‑username>/Neural_networks.git
cd Neural_networks

# Create and activate a virtual environment (recommended)
python -m venv env
env\Scripts\activate      # Windows
# or
source env/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` lists only NumPy and scikit‑learn (the latter is used solely for dataset loading and basic metrics).

---

## Basic Usage Example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from neural_nn_class import NeuralNN

# Load and prepare data
data = load_breast_cancer()
X_train, X_val, y_train, y_val = train_test_split(
    data.data, data.target.reshape(-1, 1),
    test_size=0.2, random_state=42, stratify=data.target
)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)

# Initialise and train network
net = NeuralNN(layers=[30, 16, 8, 1], lr=0.01, epochs=300, batch=32)
net.fit(X_train, y_train, X_val, y_val)

# Evaluate
y_pred  = net.predict(X_val)
y_prob  = net.predict_proba(X_val)

print("Validation accuracy :", accuracy_score(y_val, y_pred))
print("Validation ROC‑AUC  :", roc_auc_score(y_val, y_prob))
```

---

## Extending This Project

1. **Notebook Demonstration**  
   Add a Jupyter notebook that reproduces the example above, visualises the loss curve, and plots ROC and confusion‑matrix statistics.

2. **Additional Regularisation**  
   Implement an L1 penalty option and dropout between hidden layers.

3. **Early‑Stopping Callback**  
   Halt training when validation loss stops improving.

4. **Dataset Agnosticism**  
   Generalise input scaling and loss handling so the class can support multi‑class tasks or regression with minor modifications.

5. **Unit Tests and Continuous Integration**  
   Add `pytest` tests for forward/backward consistency and integrate with GitHub Actions to run tests on every commit.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
