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

## Data

Patient data (clinical spreadsheets and endoscopic images) is **not included**
in this repository for privacy reasons. Scripts expect a `DATA.xlsx` file in
their folder (and image folders for `images/`); see each folder's
`HOW_TO_RUN.md`.

## Requirements

Python 3, with pandas, numpy, scipy, scikit-learn, catboost, statsmodels, matplotlib,
and PyTorch/torchvision for the image models (`images/requirements.txt`).
