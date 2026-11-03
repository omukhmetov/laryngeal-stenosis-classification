# Laryngeal stenosis classification

Code for predicting laryngeal stenosis from clinical tabular data and
endoscopic images, and for the PICLS severity score.

| Folder | Method |
|---|---|
| `random_forest/` | Random forest classifier on clinical variables |
| `cat/` | CatBoost classifier with interpretability analysis |
| `MLP/` | Multilayer perceptron |
| `OLR/` | Ordinal logistic regression for stenosis grade |
| `PICLS_score/` | PICLS points-based risk score (odds ratios, cut-offs, decision curve) |
| `images/` | CNN backbone benchmark for healthy vs unhealthy endoscopic images |
| `fusion/` | Fusion of clinical and image models |
| `data2026/` | 2025 endoscopy set: data audit (manifest, sessions, duplicates, label conflicts) and patient-level 3-class baseline (normal / grade 1–2 / grade 3–4) with leakage and acquisition controls |

## Data

Patient data (clinical spreadsheets and endoscopic images) is **not included**
in this repository for privacy reasons. Scripts expect a `DATA.xlsx` file in
their folder (and image folders for `images/`); see each folder's
`HOW_TO_RUN.md`. `data2026/` reads the image folders and the manifest from a
path you pass on the command line (see `data2026/README.md`); the manifest
contains patient names and stays with the data.

## Requirements

Python 3, with pandas, numpy, scipy, scikit-learn, catboost, statsmodels, matplotlib,
and PyTorch/torchvision for the image models (`images/requirements.txt`);
`data2026/` also needs timm (`data2026/requirements.txt`).
