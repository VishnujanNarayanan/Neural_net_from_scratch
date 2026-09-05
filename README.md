<h1 align="center">Neural Network From Scratch — Breast Cancer Diagnosis</h1>

<p align="center">
  A 3-layer feed-forward classifier written in pure NumPy — forward pass, backprop and
  mini-batch gradient descent all hand-derived — that diagnoses breast tumours as malignant
  or benign from 30 cell-nucleus measurements at <b>97.4% accuracy</b> and <b>0.995 ROC-AUC</b>.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white"/>
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-2.2-013243?logo=numpy&logoColor=white"/>
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white"/>
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3.10-11557c?logo=plotly&logoColor=white"/>
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white"/>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-750014"/></a>
  <a href="https://github.com/VishnujanNarayanan/Neural_net_from_scratch/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/VishnujanNarayanan/Neural_net_from_scratch/actions/workflows/ci.yml/badge.svg"/></a>
  <br>
  <a href="https://vishnujannarayanan.github.io/Neural_net_from_scratch/"><img alt="Live demo" src="https://img.shields.io/badge/Live_demo-run_it_in_your_browser-2a78d6?logo=googlechrome&logoColor=white&style=for-the-badge"/></a>
  <a href="https://vishnujan-narayanan.vercel.app/"><img alt="Portfolio" src="https://img.shields.io/badge/Portfolio-vishnujan--narayanan.vercel.app-3b5998?logo=googlechrome&logoColor=white&style=for-the-badge"/></a>
  <a href="https://github.com/VishnujanNarayanan"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-VishnujanNarayanan-181717?logo=github&logoColor=white&style=for-the-badge"/></a>
  <a href="https://www.linkedin.com/in/vishnujan-narayanan"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Vishnujan_Narayanan-0A66C2?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMC40NDcgMjAuNDUyaC0zLjU1NHYtNS41NjljMC0xLjMyOC0uMDI3LTMuMDM3LTEuODUyLTMuMDM3LTEuODUzIDAtMi4xMzYgMS40NDUtMi4xMzYgMi45Mzl2NS42NjdIOS4zNTFWOWgzLjQxNHYxLjU2MWguMDQ2Yy40NzctLjkgMS42MzctMS44NSAzLjM3LTEuODUgMy42MDEgMCA0LjI2NyAyLjM3IDQuMjY3IDUuNDU1djYuMjg2ek01LjMzNyA3LjQzM2MtMS4xNDQgMC0yLjA2My0uOTI2LTIuMDYzLTIuMDY1IDAtMS4xMzguOTItMi4wNjMgMi4wNjMtMi4wNjMgMS4xNCAwIDIuMDY0LjkyNSAyLjA2NCAyLjA2MyAwIDEuMTM5LS45MjUgMi4wNjUtMi4wNjQgMi4wNjV6bTEuNzgyIDEzLjAxOUgzLjU1NVY5aDMuNTY0djExLjQ1MnpNMjIuMjI1IDBIMS43NzFDLjc5MiAwIDAgLjc3NCAwIDEuNzI5djIwLjU0MkMwIDIzLjIyNy43OTIgMjQgMS43NzEgMjRoMjAuNDUxQzIzLjIgMjQgMjQgMjMuMjI3IDI0IDIyLjI3MVYxLjcyOUMyNCAuNzc0IDIzLjIgMCAyMi4yMjIgMGguMDAzeiIvPjwvc3ZnPg%3D%3D&logoColor=white&style=for-the-badge"/></a>
  <a href="https://substack.com/@vishnujannarayanan"><img alt="Substack" src="https://img.shields.io/badge/Substack-@vishnujannarayanan-FF6719?logo=substack&logoColor=white&style=for-the-badge"/></a>
</p>

<p align="center">
  📊 <a href="#results">Results</a> ·
  🗃️ <a href="#dataset">Dataset</a> ·
  🧩 <a href="#approach">Approach</a> ·
  ⚡ <a href="#installation-and-usage">Installation</a> ·
  📁 <a href="#project-structure">Structure</a> ·
  🔬 <a href="#try-it">Live demo</a> ·
  🗄️ <a href="#querying-the-results">Queries</a> ·
  🔍 <a href="#findings">Findings</a>
</p>

---

No autograd, no Keras, no PyTorch. `NeuralNN` is 60 lines: He initialisation, ReLU hidden
layers, a sigmoid output, binary cross-entropy loss, L2 weight decay, and a manually
vectorised backward pass.

## Results

| Metric | Value |
| --- | --- |
| Validation accuracy | **97.37%** (111 / 114) |
| ROC-AUC | **0.9954** |
| Malignant recall | 97.6% (41 / 42) |
| Benign recall | 97.2% (70 / 72) |
| Trainable parameters | 641 |
| Training time | ~3 s on CPU |

<p align="center">
  <img src="figures/01_loss_curve.png" width="88%" alt="Training and validation loss over 300 epochs">
</p>
<p align="center"><i>Both curves fall together for all 300 epochs — validation ends just 0.024 above training, so the network is not overfitting.</i></p>

<p align="center">
  <img src="figures/02_roc_curve.png" width="88%" alt="ROC curve — AUC 0.995 on 114 held-out cases">
</p>
<p align="center"><i>ROC-AUC of 0.995 on 114 held-out cases — the 0.50 threshold sits in an almost empty region of the score distribution.</i></p>

<p align="center">
  <img src="figures/03_confusion_matrix.png" width="88%" alt="Confusion matrix on the 114-case validation set">
</p>
<p align="center"><i>At the default 0.50 threshold the model misses one malignant case in 42 and raises two false alarms.</i></p>

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** — 569 digitised fine-needle-aspirate images, each
described by 30 real-valued features (mean, standard error and "worst" of ten nucleus
measurements such as radius, texture, area and concavity).

| | |
| --- | --- |
| Shape | 569 rows × 30 features |
| Classes | 212 malignant (37.3%), 357 benign (62.7%) |
| Split | 455 train / 114 validation, stratified, `random_state=42` |
| Licence | CC BY 4.0 (UCI ML Repository) |

No download required — the dataset ships inside scikit-learn and is loaded with
`sklearn.datasets.load_breast_cancer()`. The original is
[UCI dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## Approach

1. **Load and split.** 569 samples split 80/20, stratified on the diagnosis so both sets keep
   the 37/63 class balance.
2. **Standardise.** Raw features span four orders of magnitude — `mean area` runs 143–2501
   while `mean smoothness` runs 0.053–0.163. A `StandardScaler` is fitted on the training
   split *only* and applied to both, so no validation statistics leak into training.
3. **Build the network.** Layers `[30, 16, 8, 1]`. Weights use He initialisation
   (`randn * sqrt(2/fan_in)`), correct for ReLU. Hidden layers are ReLU, the output is a
   sigmoid producing P(benign).
4. **Train.** 300 epochs of mini-batch gradient descent, batch 32, learning rate 0.01, L2
   weight decay 1e-4, reshuffled every epoch. Backprop is derived by hand and fully
   vectorised — the sigmoid-with-BCE output layer collapses to `dZ = A - y`.
5. **Evaluate.** Accuracy, ROC-AUC and a confusion matrix on the held-out split, plus a
   threshold sweep to test whether the one missed malignancy is recoverable.

## Installation and usage

```bash
git clone https://github.com/VishnujanNarayanan/Neural_net_from_scratch.git
cd Neural_net_from_scratch

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Run the analysis end to end and regenerate every figure in `figures/`:

```bash
jupyter nbconvert --to notebook --execute --inplace NeuralNet_BreastCancer.ipynb
```

Or open it interactively:

```bash
jupyter lab NeuralNet_BreastCancer.ipynb
```

Both write `figures/01_loss_curve.png`, `02_roc_curve.png` and `03_confusion_matrix.png` at
1600×1000. All figure styling comes from `viz_style.py`; notebooks never set colours or fonts.

## Project structure

```
Neural_net_from_scratch/
├── NeuralNet_BreastCancer.ipynb   # The project: NeuralNN class, training, evaluation, figures
├── viz_style.py                   # Shared figure style — palette, type scale, layout, save path
├── figures/                       # Generated PNGs, 1600x1000, overwritten on each run
│   ├── 01_loss_curve.png
│   ├── 02_roc_curve.png
│   └── 03_confusion_matrix.png
├── db/
│   ├── queries.sql                # Named queries behind every claim in Findings
│   └── results.db                 # SQLite, rebuilt on each run (gitignored)
├── model/
│   └── weights.npz                # Exported weights, biases and scaler stats — 641 parameters
├── docs/                          # Static browser demo — GitHub Pages, no server, no cold start
│   ├── index.html                 # The forward pass, reimplemented in JavaScript
│   ├── model.json                 # 641 parameters + scaler stats
│   └── cases.json                 # The 569 research cases
├── app/                           # Gradio inference demo, for local use or a Gradio Space
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── scripts/
│   ├── check_model.py             # CI guard: re-scores the validation split, fails under 95%
│   └── export_web.py              # Regenerates docs/*.json from model/weights.npz
├── .github/workflows/ci.yml       # Executes the notebook on every push
├── requirements.txt
├── LICENSE
└── README.md
```

## Try it

**[Run it in your browser →](https://vishnujannarayanan.github.io/Neural_net_from_scratch/)** — no install, no sign-up, and **no cold start**.

All 641 parameters run client-side in JavaScript, so there is no server to wake: the page
loads `model.json` and `cases.json` (138 KB together) and does the same three matrix
multiplies the notebook does.

**Drag the decision line.** The network outputs a score, not a verdict, and turning one
into the other means choosing a cut-off. Every held-out case is a dot; move the line and
watch missed cancers trade against false alarms. At 0.50 the model misses one cancer and
raises two false alarms; push to 0.80 and it catches every cancer at the cost of thirteen.
That tradeoff is the whole point — the cut-off is a clinical decision, not a modelling one.

You can also inspect any of the 569 individual cases and see the measurements the network
scored.

Also mirrored as a Hugging Face Space: **[Vishnujann/nn-from-scratch-breast-cancer](https://huggingface.co/spaces/Vishnujann/nn-from-scratch-breast-cancer)**.

> **This is a learning project, not a medical device.** It is trained on 455 cases from a
> single 1990s research dataset, has no clinical validation of any kind, and must not be
> used to make any decision about anyone's health.

`docs/` is generated by `scripts/export_web.py` from `model/weights.npz`, and CI fails if
the two drift apart, so the page can never show stale weights.

### The Gradio version

`app/` is the same model behind a [Gradio](https://gradio.app) interface, kept for local
use and for deploying to a Gradio Space:

```bash
pip install -r app/requirements.txt
python app/app.py                 # http://127.0.0.1:7860
```

Note that Hugging Face now requires a PRO subscription to host Gradio Spaces on CPU —
static Spaces, like the mirror above, remain free.

## Querying the results

After a run, `db/results.db` holds two tables — `cases` (all 569 cases, their 30
measurements, true diagnosis and split) and `predictions` (the model's `p_benign` and
assigned label for each validation case). `db/queries.sql` holds the five named queries
the notebook executes and prints:

| query | question it answers |
| --- | --- |
| `class_balance` | Did the stratified split preserve the 37/63 ratio? |
| `feature_ranges` | Which measurements span the widest raw range? |
| `misclassified` | Which validation cases were wrong, and how confidently? |
| `score_separation` | How far apart are the two classes' scores? |
| `threshold_sweep` | Can a different decision threshold recover the missed malignancy? |

Every number in [Findings](#findings) comes from these queries rather than from prose, so
re-running the notebook re-derives them:

```bash
sqlite3 db/results.db < db/queries.sql
```

## Findings

- **A 641-parameter network is enough.** Hand-written backprop reaches 97.37% accuracy and
  0.9954 ROC-AUC on 114 held-out cases — within noise of scikit-learn's tuned estimators on
  this dataset. The problem is close to linearly separable once features are standardised.
- **The model is underfitted, not overfitted.** After 300 epochs validation loss is still
  falling and sits only 0.024 above training loss. The usual portfolio failure mode —
  memorising 455 rows with a curve that turns upward — never appears; more epochs or more
  capacity would likely still help.
- **The one missed malignancy is not a borderline call.** It scores P(benign) = 0.732, above
  the 10th percentile of genuinely benign cases (0.730). Raising the threshold to 0.60 or
  0.70 does not recover it; only 0.80 does, and that turns 2 false alarms into 13 while
  dropping accuracy to 88.6%. This case is confidently wrong, so threshold tuning is the
  wrong lever — it needs better features or more data.
- **Separation is near-total elsewhere.** Median P(benign) is 0.012 for malignant cases and
  0.953 for benign ones. The 0.50 threshold sits in an almost empty region, which is why
  accuracy is stable across 0.30–0.60.
- **Standardisation is doing real work.** With `mean area` (143–2501) and `mean smoothness`
  (0.053–0.163) on the same input layer, unscaled gradients are dominated by the
  large-magnitude columns and training stalls.

## Author

<p align="center">
  <strong>Vishnujan Narayanan</strong>
</p>

<p align="center">
  <a href="https://vishnujan-narayanan.vercel.app/"><img alt="Portfolio" src="https://img.shields.io/badge/Portfolio-vishnujan--narayanan.vercel.app-3b5998?logo=googlechrome&logoColor=white&style=for-the-badge"/></a>
  <a href="https://github.com/VishnujanNarayanan"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-VishnujanNarayanan-181717?logo=github&logoColor=white&style=for-the-badge"/></a>
  <a href="https://www.linkedin.com/in/vishnujan-narayanan"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Vishnujan_Narayanan-0A66C2?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMC40NDcgMjAuNDUyaC0zLjU1NHYtNS41NjljMC0xLjMyOC0uMDI3LTMuMDM3LTEuODUyLTMuMDM3LTEuODUzIDAtMi4xMzYgMS40NDUtMi4xMzYgMi45Mzl2NS42NjdIOS4zNTFWOWgzLjQxNHYxLjU2MWguMDQ2Yy40NzctLjkgMS42MzctMS44NSAzLjM3LTEuODUgMy42MDEgMCA0LjI2NyAyLjM3IDQuMjY3IDUuNDU1djYuMjg2ek01LjMzNyA3LjQzM2MtMS4xNDQgMC0yLjA2My0uOTI2LTIuMDYzLTIuMDY1IDAtMS4xMzguOTItMi4wNjMgMi4wNjMtMi4wNjMgMS4xNCAwIDIuMDY0LjkyNSAyLjA2NCAyLjA2MyAwIDEuMTM5LS45MjUgMi4wNjUtMi4wNjQgMi4wNjV6bTEuNzgyIDEzLjAxOUgzLjU1NVY5aDMuNTY0djExLjQ1MnpNMjIuMjI1IDBIMS43NzFDLjc5MiAwIDAgLjc3NCAwIDEuNzI5djIwLjU0MkMwIDIzLjIyNy43OTIgMjQgMS43NzEgMjRoMjAuNDUxQzIzLjIgMjQgMjQgMjMuMjI3IDI0IDIyLjI3MVYxLjcyOUMyNCAuNzc0IDIzLjIgMCAyMi4yMjIgMGguMDAzeiIvPjwvc3ZnPg%3D%3D&logoColor=white&style=for-the-badge"/></a>
  <a href="https://substack.com/@vishnujannarayanan"><img alt="Substack" src="https://img.shields.io/badge/Substack-@vishnujannarayanan-FF6719?logo=substack&logoColor=white&style=for-the-badge"/></a>
</p>

## Licence

Released under the MIT Licence — see [LICENSE](LICENSE).
