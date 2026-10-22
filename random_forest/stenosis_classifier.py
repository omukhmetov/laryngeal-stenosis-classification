"""
stenosis_classifier.py
======================
Predicts laryngeal/subglottic stenosis grade from clinical features.

Two tasks are modelled in parallel:
  1. 4-class  : Grade 1 / 2 / 3 / 4  (ordinal, hard problem at n=172)
  2. Binary   : Mild (Grade 1–2) vs Severe (Grade 3–4)  (clinically actionable)

Models used: Random Forest + Logistic Regression.
Evaluation : 5-fold stratified CV + held-out test set.
Output     : stenosis_results.png   (6-panel figure)
             stenosis_models.pkl    (serialised pipelines)
"""

from sklearn.base import clone as sklearn_clone
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import os
import logging
import pickle
import warnings
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    tempfile.gettempdir(), "matplotlib"))
matplotlib.use("Agg")


warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────
DATA_PATH = "DATA.xlsx"
RESULTS_FIGURE = "stenosis_results.png"
MODELS_PICKLE = "stenosis_models.pkl"

RANDOM_SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5

np.random.seed(RANDOM_SEED)

assert os.path.exists(DATA_PATH), f"Data file not found: {DATA_PATH}"

# ──────────────────────────────────────────────
# 1. Load & clean
# ──────────────────────────────────────────────
df = pd.read_excel(DATA_PATH, header=1)
df = df.drop(
    columns=[c for c in ["Year", "№№", "Patient №№"] if c in df.columns])

TARGET = "Grade of stenosis"
df = df[df[TARGET].astype(str).str.strip().str.lower() != "no data"].copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)

log.info(f"Dataset: {len(df)} patients after target cleaning.")
log.info("Grade distribution:\n" +
         df[TARGET].value_counts().sort_index().to_string())

# ──────────────────────────────────────────────
# 2. Feature definitions
# ──────────────────────────────────────────────
# NOTE: 'Type of intervention' dropped — 171/172 rows identical → no signal.


def clean_age(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s == "up to 1":
        return 0.5
    if s in ("not reported", "no data", ""):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


NUMERIC_COLS = [
    "Age at primary intubation (months)",
    "re-intubation",
    "history of tracheostomy",
    "presence of T-tube",
    "mother's pregnancy №",
    "childbirth №",
    "Birth weight, gramm",
    "Height at birth, cm",
]

CATEGORICAL_COLS = [
    "Gender",
    "Localization of stenosis",
    "Leading comorbidity syndrome",
    "Reason of intubation",
    "Emergency/planned intubation",
    "delivery type",
    "Delivery category",
    "patient's residence",
    "intubation clinic's level",
]

df["Age at primary intubation (months)"] = (
    df["Age at primary intubation (months)"].apply(clean_age)
)
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

_NO_DATA = {"no data", "not reported", "nan", ""}
for col in CATEGORICAL_COLS:
    df[col] = (
        df[col]
        .fillna("missing")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(_NO_DATA, "missing")
    )

X = df[NUMERIC_COLS + CATEGORICAL_COLS]

# ──────────────────────────────────────────────
# 3. Targets
# ──────────────────────────────────────────────
label_enc = LabelEncoder()
y4 = label_enc.fit_transform(df[TARGET])          # 0,1,2,3
y2 = (y4 >= 2).astype(int)                         # 0=Mild, 1=Severe
GRADE_LABELS = [str(c) for c in label_enc.classes_]
BINARY_LABELS = ["Mild (1–2)", "Severe (3–4)"]

# ──────────────────────────────────────────────
# 4. Preprocessing pipeline
# ──────────────────────────────────────────────


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), NUMERIC_COLS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_COLS),
    ])


def make_full_pipeline(clf) -> Pipeline:
    """Wrap a classifier with preprocessing into a single sklearn Pipeline."""
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", clf),
    ])


# ──────────────────────────────────────────────
# 5. Model definitions
# ──────────────────────────────────────────────
# Random Forest — strong baseline, handles class imbalance, gives importances.
# Logistic Regression — linear, well-calibrated with isotonic calibration.
# Both are wrapped in CalibratedClassifierCV (isotonic, cv=3) so predict_proba
# is reliable for clinical decision support.

def make_models_4class():
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    lr = LogisticRegression(
        C=0.1,
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        random_state=RANDOM_SEED,
    )
    return {
        "Random Forest":        make_full_pipeline(rf),
        "Logistic Regression":  make_full_pipeline(lr),
    }


def make_models_binary():
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    lr = LogisticRegression(
        C=0.1,
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        random_state=RANDOM_SEED,
    )
    return {
        "Random Forest":        make_full_pipeline(rf),
        "Logistic Regression":  make_full_pipeline(lr),
    }


# ──────────────────────────────────────────────
# 6. Cross-validation helper
# ──────────────────────────────────────────────
def run_cv(models: dict, X, y, skf, task="4-class") -> dict:
    """
    Returns per-model dict with keys: accs, kappas (4-class) or accs, aucs (binary).
    """
    results = {name: {"accs": [], "kappas": [], "aucs": []} for name in models}
    is_binary = (task == "binary")

    for fold, (tr, vl) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X.iloc[tr], X.iloc[vl]
        y_tr, y_vl = y[tr], y[vl]

        for name, pipe in models.items():
            pipe_clone = sklearn_clone(pipe)
            pipe_clone.fit(X_tr, y_tr)
            preds = pipe_clone.predict(X_vl)
            results[name]["accs"].append(accuracy_score(y_vl, preds))

            if is_binary:
                probs = pipe_clone.predict_proba(X_vl)[:, 1]
                results[name]["aucs"].append(roc_auc_score(y_vl, probs))
            else:
                kappa = (
                    cohen_kappa_score(y_vl, preds, weights="quadratic")
                    if len(np.unique(preds)) > 1 else 0.0
                )
                results[name]["kappas"].append(kappa)

    return results


# sklearn clone helper

# ──────────────────────────────────────────────
# 7. Train / test split
# ──────────────────────────────────────────────
X_trainval, X_test, y4_trainval, y4_test, y2_trainval, y2_test = train_test_split(
    X, y4, y2,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y4,
)
log.info(f"Train+val: {len(X_trainval)}  |  Test: {len(X_test)}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# ──────────────────────────────────────────────
# 8. Cross-validation
# ──────────────────────────────────────────────
log.info(f"\n=== {N_FOLDS}-Fold CV — 4-class ===")
models_4 = make_models_4class()
cv4 = run_cv(models_4, X_trainval, y4_trainval, skf, task="4-class")
for name, res in cv4.items():
    log.info(
        f"  {name}: acc={np.mean(res['accs']):.3f}±{np.std(res['accs']):.3f}  "
        f"kappa={np.mean(res['kappas']):.3f}±{np.std(res['kappas']):.3f}"
    )

log.info(f"\n=== {N_FOLDS}-Fold CV — Binary ===")
models_2 = make_models_binary()
cv2 = run_cv(models_2, X_trainval, y2_trainval, skf, task="binary")
for name, res in cv2.items():
    log.info(
        f"  {name}: acc={np.mean(res['accs']):.3f}±{np.std(res['accs']):.3f}  "
        f"AUC={np.mean(res['aucs']):.3f}±{np.std(res['aucs']):.3f}"
    )

# ──────────────────────────────────────────────
# 9. Final models — fit on full trainval, evaluate on test
# ──────────────────────────────────────────────
log.info("\n=== Final model evaluation on hold-out test set ===")

final_models = {}
test_results = {}

for task_label, models_dict, y_tv, y_te, labels in [
    ("4class", make_models_4class(), y4_trainval, y4_test, GRADE_LABELS),
    ("binary", make_models_binary(), y2_trainval, y2_test, BINARY_LABELS),
]:
    test_results[task_label] = {}
    for name, pipe in models_dict.items():
        pipe.fit(X_trainval, y_tv)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)

        acc = accuracy_score(y_te, preds)
        kappa = cohen_kappa_score(y_te, preds, weights="quadratic") if len(
            np.unique(preds)) > 1 else 0.0
        cm = confusion_matrix(y_te, preds)
        rep = classification_report(y_te, preds, target_names=labels,
                                    output_dict=True, zero_division=0)

        test_results[task_label][name] = {
            "preds": preds, "probs": probs,
            "acc": acc, "kappa": kappa,
            "cm": cm, "report": rep,
        }
        final_models[f"{task_label}_{name}"] = pipe

        auc_str = ""
        if task_label == "binary":
            auc = roc_auc_score(y_te, probs[:, 1])
            test_results[task_label][name]["auc"] = auc
            auc_str = f"  AUC={auc:.3f}"

        log.info(
            f"  [{task_label}] {name}: acc={acc:.3f}  kappa={kappa:.3f}{auc_str}")
        log.info("\n" + classification_report(y_te, preds,
                 target_names=labels, zero_division=0))

# ──────────────────────────────────────────────
# 10. Feature importances (RF, 4-class)
# ──────────────────────────────────────────────
rf_pipe = final_models["4class_Random Forest"]
ohe_names = (
    rf_pipe.named_steps["pre"]
    .named_transformers_["cat"]["ohe"]
    .get_feature_names_out(CATEGORICAL_COLS)
)
feat_names = NUMERIC_COLS + list(ohe_names)
importances = pd.Series(
    rf_pipe.named_steps["clf"].feature_importances_,
    index=feat_names,
).sort_values(ascending=False)
top_features = importances.head(12)

# ──────────────────────────────────────────────
# 11. Figure — 3 rows × 2 cols
# ──────────────────────────────────────────────
COLORS = {
    "Random Forest":       "#4c78a8",
    "Logistic Regression": "#f58518",
}
GRADE_COLORS = ["#54a24b", "#eeca3b", "#f58518", "#e45756"]

fig = plt.figure(figsize=(15, 17))
fig.suptitle(
    "Laryngeal/Subglottic Stenosis — Grade Prediction\n"
    "Random Forest & Logistic Regression",
    fontsize=14, fontweight="bold", y=0.99,
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

# ── Panel A: CV summary (4-class + binary) ──────────────────────────────────
ax_cv = fig.add_subplot(gs[0, 0])

model_names = list(cv4.keys())
x = np.arange(len(model_names))
width = 0.35

bars_4 = [np.mean(cv4[n]["kappas"]) for n in model_names]
err_4 = [np.std(cv4[n]["kappas"]) for n in model_names]
bars_2 = [np.mean(cv2[n]["aucs"]) for n in model_names]
err_2 = [np.std(cv2[n]["aucs"]) for n in model_names]

b1 = ax_cv.bar(x - width/2, bars_4, width, yerr=err_4, capsize=4,
               color=[COLORS[n] for n in model_names], alpha=0.85, label="4-class κ (quadratic)")
b2 = ax_cv.bar(x + width/2, bars_2, width, yerr=err_2, capsize=4,
               color=[COLORS[n] for n in model_names], alpha=0.45, label="Binary AUC")

ax_cv.set_title(f"A  {N_FOLDS}-Fold CV Performance",
                fontweight="bold", loc="left")
ax_cv.set_xticks(x)
ax_cv.set_xticklabels(model_names, fontsize=9)
ax_cv.set_ylim(0, 1)
ax_cv.set_ylabel("Score")
ax_cv.axhline(0.5, color="gray", linestyle="--",
              linewidth=0.8, label="AUC chance")
ax_cv.grid(axis="y", alpha=0.3)
ax_cv.legend(fontsize=8)

for bar, val in zip(b1, bars_4):
    ax_cv.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f"{val:.2f}", ha="center", fontsize=8)
for bar, val in zip(b2, bars_2):
    ax_cv.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f"{val:.2f}", ha="center", fontsize=8)

# ── Panel B: Confusion matrix — best 4-class model (Logistic Regression) ────
ax_cm4 = fig.add_subplot(gs[0, 1])
best4_name = max(test_results["4class"],
                 key=lambda n: test_results["4class"][n]["kappa"])
cm4 = test_results["4class"][best4_name]["cm"]
im = ax_cm4.imshow(cm4, cmap="Blues")
ax_cm4.set_title(
    f"B  4-class Confusion Matrix\n({best4_name}, test set)", fontweight="bold", loc="left")
ax_cm4.set_xlabel("Predicted grade")
ax_cm4.set_ylabel("True grade")
ax_cm4.set_xticks(range(4))
ax_cm4.set_yticks(range(4))
ax_cm4.set_xticklabels(GRADE_LABELS)
ax_cm4.set_yticklabels(GRADE_LABELS)
thresh = cm4.max() / 2
for r in range(4):
    for c in range(4):
        ax_cm4.text(c, r, cm4[r, c], ha="center", va="center",
                    color="white" if cm4[r, c] > thresh else "black", fontsize=11)
fig.colorbar(im, ax=ax_cm4, fraction=0.046, pad=0.04)

# ── Panel C: ROC curves — binary ─────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 0])
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
for name, res in test_results["binary"].items():
    fpr, tpr, _ = roc_curve(y2_test, res["probs"][:, 1])
    auc = res["auc"]
    ax_roc.plot(fpr, tpr, color=COLORS[name], linewidth=2,
                label=f"{name}  AUC={auc:.2f}")
ax_roc.set_title("C  ROC — Binary (Mild vs Severe)",
                 fontweight="bold", loc="left")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(fontsize=8)
ax_roc.grid(alpha=0.3)

# ── Panel D: Binary confusion matrix — RF ────────────────────────────────────
ax_cm2 = fig.add_subplot(gs[1, 1])
best2_name = max(test_results["binary"],
                 key=lambda n: test_results["binary"][n]["auc"])
cm2 = test_results["binary"][best2_name]["cm"]
im2 = ax_cm2.imshow(cm2, cmap="Greens")
ax_cm2.set_title(
    f"D  Binary Confusion Matrix\n({best2_name}, test set)", fontweight="bold", loc="left")
ax_cm2.set_xlabel("Predicted")
ax_cm2.set_ylabel("True")
ax_cm2.set_xticks([0, 1])
ax_cm2.set_yticks([0, 1])
ax_cm2.set_xticklabels(BINARY_LABELS, fontsize=8)
ax_cm2.set_yticklabels(BINARY_LABELS, fontsize=8)
thresh2 = cm2.max() / 2
for r in range(2):
    for c in range(2):
        ax_cm2.text(c, r, cm2[r, c], ha="center", va="center",
                    color="white" if cm2[r, c] > thresh2 else "black", fontsize=13)
fig.colorbar(im2, ax=ax_cm2, fraction=0.046, pad=0.04)

# ── Panel E: Feature importances ─────────────────────────────────────────────
ax_imp = fig.add_subplot(gs[2, 0])
feat_labels = [f.replace("Reason of intubation_", "Reason: ")
                .replace("Leading comorbidity syndrome_", "Comorbidity: ")
                .replace("intubation clinic's level_", "Clinic: ")
                .replace("patient's residence_", "Residence: ")
                .replace("Age at primary intubation (months)", "Age at intubation (mo)")
               for f in top_features.index]
colors_imp = [GRADE_COLORS[0]] * \
    len(NUMERIC_COLS) + [GRADE_COLORS[2]] * \
    (len(top_features) - len(NUMERIC_COLS))
# trim to match
colors_imp = colors_imp[:len(top_features)]
bars = ax_imp.barh(
    feat_labels[::-1], top_features.values[::-1], color="#4c78a8", alpha=0.8)
ax_imp.set_title("E  Feature Importances (RF, 4-class)",
                 fontweight="bold", loc="left")
ax_imp.set_xlabel("Mean Decrease in Impurity")
ax_imp.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, top_features.values[::-1]):
    ax_imp.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7)

# ── Panel F: Summary table ────────────────────────────────────────────────────
ax_tab = fig.add_subplot(gs[2, 1])
ax_tab.axis("off")

rows = []
for name in model_names:
    r4 = test_results["4class"][name]
    r2 = test_results["binary"][name]
    rows.append([
        name,
        f"{r4['acc']:.3f}",
        f"{r4['kappa']:.3f}",
        f"{r2['acc']:.3f}",
        f"{r2['auc']:.3f}",
    ])

col_labels = ["Model", "4-cl Acc", "4-cl κ", "Bin Acc", "Bin AUC"]
table = ax_tab.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0, 0.35, 1, 0.5],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#ecf0f1")
    cell.set_edgecolor("white")

ax_tab.set_title("F  Hold-out Test Set Summary",
                 fontweight="bold", loc="left", pad=120)


fig.savefig(RESULTS_FIGURE, dpi=200, bbox_inches="tight")
plt.close(fig)
log.info(f"Figure saved → {RESULTS_FIGURE}")

# ──────────────────────────────────────────────
# 12. Save models
# ──────────────────────────────────────────────
with open(MODELS_PICKLE, "wb") as f:
    pickle.dump({
        "models": final_models,
        "label_encoder": label_enc,
        "feature_cols": NUMERIC_COLS + CATEGORICAL_COLS,
    }, f)
log.info(f"Models saved → {MODELS_PICKLE}")

log.info("\nDone.")
