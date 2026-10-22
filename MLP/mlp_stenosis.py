from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, cohen_kappa_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib
import os
import tempfile
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    tempfile.gettempdir(), "matplotlib"))
matplotlib.use("Agg")


# =========================
# Logging
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# =========================
# Settings
# =========================

DATA_PATH = "DATA.xlsx"
RESULTS_FIGURE_PATH = "stenosis_results.png"
MODEL_SAVE_PATH = "stenosis_mlp_model.pth"

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5          # k-fold CV on train set
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MIN_DELTA = 0.001

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

assert os.path.exists(DATA_PATH), f"Data file not found: {DATA_PATH}"

# =========================
# Load data
# =========================

df = pd.read_excel(DATA_PATH, header=1)
df = df.drop(
    columns=[c for c in ["Year", "№№", "Patient №№"] if c in df.columns])

target_col = "Grade of stenosis"

# Clean target
df = df[df[target_col].astype(str).str.strip().str.lower() != "no data"].copy()
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

log.info(f"Loaded {len(df)} rows after cleaning target.")

# =========================
# Feature selection
# NOTE: 'Type of intervention' is dropped — 171/172 rows are identical ('intubation'),
#       making it a near-constant feature that would not generalize.
# =========================


def clean_age(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s == "up to 1":
        return 0.5
    if s in ["not reported", "no data", ""]:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


age_col = "Age at primary intubation (months)"
df[age_col] = df[age_col].apply(clean_age)

numeric_cols = [
    "Age at primary intubation (months)",
    "re-intubation",
    "history of tracheostomy",
    "presence of T-tube",
    "mother's pregnancy №",
    "childbirth №",
    "Birth weight, gramm",
    "Height at birth, cm",
]

categorical_cols = [
    "Gender",
    # "Type of intervention" — DROPPED: near-constant (171/172 = 'intubation')
    "Localization of stenosis",
    "Leading comorbidity syndrome",
    "Reason of intubation",
    "Emergency/planned intubation",
    "delivery type",
    "Delivery category",
    "patient's residence",
    "intubation clinic's level",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Treat "no data" strings as missing in categoricals so they get imputed,
# not one-hot encoded as a separate spurious category.
NO_DATA_STRINGS = {"no data", "not reported", "nan", ""}
for col in categorical_cols:
    df[col] = (
        df[col]
        .fillna("missing")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(NO_DATA_STRINGS, "missing")
    )

X = df[numeric_cols + categorical_cols]
y_raw = df[target_col]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

log.info(f"Classes: {label_encoder.classes_}")
log.info(
    f"Class distribution:\n{pd.Series(y_raw).value_counts().sort_index().to_string()}")

# =========================
# Hold-out test split
# =========================

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

log.info(f"Hold-out test set: {len(X_test)} samples")
log.info(f"Train+val pool:    {len(X_trainval)} samples")

# =========================
# Preprocessing pipeline builder
# =========================


def build_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])


def to_dense(arr):
    return arr.toarray() if hasattr(arr, "toarray") else arr


# =========================
# Ordinal loss helper
# Penalises predictions proportionally to how far they are from the true grade.
# Grade 1 predicted as 4 is penalised more than grade 1 predicted as 2.
# =========================

def ordinal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    indices = torch.arange(
        num_classes, dtype=torch.float32, device=logits.device)
    # dist_matrix[i, j] = (i - j)^2 — squared distance between grade i and j
    dist_matrix = (indices.unsqueeze(0) - indices.unsqueeze(1)) ** 2
    weights = dist_matrix[targets]         # (batch, num_classes)
    return (probs * weights).sum(dim=1).mean()


# =========================
# Model definition
# =========================

class StenosisMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =========================
# Train one fold / final model
# =========================

def train_model(
    X_tr, y_tr, X_vl, y_vl,
    input_dim, num_classes,
    class_weights_tensor,
    fold_label="",
):
    model = StenosisMLP(input_dim, num_classes)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = ce_criterion(logits, batch_y) + 0.5 * \
                ordinal_loss(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_y)
        avg_train_loss = epoch_loss / len(y_tr)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl)
            val_loss = (
                ce_criterion(val_logits, y_vl)
                + 0.5 * ordinal_loss(val_logits, y_vl)
            ).item()

        scheduler.step(val_loss)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            log.info(
                f"  {fold_label}Early stop at epoch {epoch+1} (best={best_epoch}).")
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch


# =========================
# k-Fold cross-validation
# =========================

log.info(f"\n=== {N_FOLDS}-Fold Cross-Validation ===")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

fold_accuracies = []
fold_kappas = []
all_fold_train_losses = []
all_fold_val_losses = []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_trainval, y_trainval), start=1):
    X_tr_raw = X_trainval.iloc[tr_idx]
    X_vl_raw = X_trainval.iloc[vl_idx]
    y_tr = y_trainval[tr_idx]
    y_vl = y_trainval[vl_idx]

    preprocessor = build_preprocessor()
    X_tr_proc = to_dense(preprocessor.fit_transform(X_tr_raw))
    X_vl_proc = to_dense(preprocessor.transform(X_vl_raw))

    X_tr_t = torch.tensor(X_tr_proc, dtype=torch.float32)
    X_vl_t = torch.tensor(X_vl_proc, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_vl_t = torch.tensor(y_vl, dtype=torch.long)

    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_tensor = torch.tensor(cw, dtype=torch.float32)

    input_dim = X_tr_t.shape[1]
    num_classes = len(label_encoder.classes_)

    model, tr_losses, vl_losses, best_ep = train_model(
        X_tr_t, y_tr_t, X_vl_t, y_vl_t,
        input_dim, num_classes, cw_tensor,
        fold_label=f"[Fold {fold}] "
    )

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_vl_t), dim=1).numpy()

    acc = accuracy_score(y_vl, preds)
    kappa = cohen_kappa_score(y_vl, preds, weights="quadratic")
    fold_accuracies.append(acc)
    fold_kappas.append(kappa)
    all_fold_train_losses.append(tr_losses)
    all_fold_val_losses.append(vl_losses)

    log.info(f"  Fold {fold}: accuracy={acc:.3f}, kappa={kappa:.3f}")

log.info(
    f"\nCV accuracy: {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}")
log.info(
    f"CV kappa:    {np.mean(fold_kappas):.3f} ± {np.std(fold_kappas):.3f}")

# =========================
# Final model — train on full trainval, evaluate on hold-out test
# =========================

log.info("\n=== Training final model on full train+val set ===")

final_preprocessor = build_preprocessor()
X_trainval_proc = to_dense(final_preprocessor.fit_transform(X_trainval))
X_test_proc = to_dense(final_preprocessor.transform(X_test))

X_trainval_t = torch.tensor(X_trainval_proc, dtype=torch.float32)
X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)
y_trainval_t = torch.tensor(y_trainval, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Small internal validation split just for early stopping on the final model
X_ftrain, X_fval, y_ftrain, y_fval = train_test_split(
    X_trainval_t, y_trainval_t,
    test_size=0.15,
    random_state=RANDOM_SEED,
    stratify=y_trainval_t,
)

cw_final = compute_class_weight(
    "balanced", classes=np.unique(y_trainval), y=y_trainval)
cw_final_tensor = torch.tensor(cw_final, dtype=torch.float32)

input_dim = X_trainval_t.shape[1]
num_classes = len(label_encoder.classes_)

log.info(f"Input dimension after encoding: {input_dim}")

final_model, final_tr_losses, final_vl_losses, final_best_ep = train_model(
    X_ftrain, y_ftrain, X_fval, y_fval,
    input_dim, num_classes, cw_final_tensor,
    fold_label="[Final] "
)

# =========================
# Test set evaluation
# =========================

final_model.eval()
with torch.no_grad():
    test_logits = final_model(X_test_t)
    test_preds = torch.argmax(test_logits, dim=1).numpy()

class_labels = [str(c) for c in label_encoder.classes_]
test_accuracy = accuracy_score(y_test, test_preds)
weighted_kappa = cohen_kappa_score(y_test, test_preds, weights="quadratic")
report_dict = classification_report(
    y_test, test_preds,
    target_names=class_labels,
    output_dict=True,
    zero_division=0,
)
conf_matrix = confusion_matrix(y_test, test_preds)

log.info(f"\nTest Accuracy:         {test_accuracy:.4f}")
log.info(f"Quadratic Weighted κ:  {weighted_kappa:.4f}")
log.info(f"\nClassification Report:\n"
         + classification_report(y_test, test_preds, target_names=class_labels, zero_division=0))
log.info(f"Confusion Matrix:\n{conf_matrix}")

# =========================
# Results figure (3 rows × 2 cols)
# =========================

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle("Stenosis MLP Results", fontsize=16, fontweight="bold")

# --- (0,0) Final model training curve ---
epochs_final = list(range(1, len(final_tr_losses) + 1))
axes[0, 0].plot(epochs_final, final_tr_losses, label="Train loss", linewidth=2)
axes[0, 0].plot(epochs_final, final_vl_losses,
                label="Val loss (ES)", linewidth=2)
axes[0, 0].axvline(final_best_ep, color="gray", linestyle="--", linewidth=1,
                   label=f"Best epoch {final_best_ep}")
axes[0, 0].set_title("Final Model — Loss During Training")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()

# --- (0,1) Confusion matrix ---
matrix_plot = axes[0, 1].imshow(conf_matrix, cmap="Blues")
axes[0, 1].set_title("Confusion Matrix (Hold-out Test)")
axes[0, 1].set_xlabel("Predicted grade")
axes[0, 1].set_ylabel("True grade")
axes[0, 1].set_xticks(np.arange(len(class_labels)))
axes[0, 1].set_yticks(np.arange(len(class_labels)))
axes[0, 1].set_xticklabels(class_labels)
axes[0, 1].set_yticklabels(class_labels)
threshold = conf_matrix.max() / 2 if conf_matrix.max() > 0 else 1
for r in range(conf_matrix.shape[0]):
    for c in range(conf_matrix.shape[1]):
        color = "white" if conf_matrix[r, c] > threshold else "black"
        axes[0, 1].text(c, r, conf_matrix[r, c], ha="center",
                        va="center", color=color)
fig.colorbar(matrix_plot, ax=axes[0, 1], fraction=0.046, pad=0.04)

# --- (1,0) Per-class metrics ---
x = np.arange(len(class_labels))
width = 0.25
for offset, metric in enumerate(["precision", "recall", "f1-score"]):
    values = [report_dict[label][metric] for label in class_labels]
    axes[1, 0].bar(x + (offset - 1) * width, values, width, label=metric)
axes[1, 0].set_title("Per-class Metrics (Test Set)")
axes[1, 0].set_xlabel("Grade of stenosis")
axes[1, 0].set_ylabel("Score")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(class_labels)
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(axis="y", alpha=0.3)
axes[1, 0].legend()

# --- (1,1) Overall test metrics ---
summary_labels = ["Accuracy", "Weighted κ"]
summary_values = [test_accuracy, weighted_kappa]
bar_colors = ["#4c78a8", "#f58518"]
axes[1, 1].bar(summary_labels, summary_values, color=bar_colors)
axes[1, 1].set_title("Overall Test Metrics")
axes[1, 1].set_ylim(0, 1)
axes[1, 1].grid(axis="y", alpha=0.3)
for i, v in enumerate(summary_values):
    axes[1, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

# --- (2,0) CV accuracy per fold ---
fold_nums = list(range(1, N_FOLDS + 1))
axes[2, 0].bar(fold_nums, fold_accuracies, color="#72b7b2")
axes[2, 0].axhline(np.mean(fold_accuracies), color="red", linestyle="--",
                   label=f"Mean={np.mean(fold_accuracies):.3f}")
axes[2, 0].set_title(f"{N_FOLDS}-Fold CV Accuracy")
axes[2, 0].set_xlabel("Fold")
axes[2, 0].set_ylabel("Accuracy")
axes[2, 0].set_ylim(0, 1)
axes[2, 0].set_xticks(fold_nums)
axes[2, 0].grid(axis="y", alpha=0.3)
axes[2, 0].legend()

# --- (2,1) CV kappa per fold ---
axes[2, 1].bar(fold_nums, fold_kappas, color="#54a24b")
axes[2, 1].axhline(np.mean(fold_kappas), color="red", linestyle="--",
                   label=f"Mean={np.mean(fold_kappas):.3f}")
axes[2, 1].set_title(f"{N_FOLDS}-Fold CV Quadratic Weighted κ")
axes[2, 1].set_xlabel("Fold")
axes[2, 1].set_ylabel("Kappa")
axes[2, 1].set_ylim(0, 1)
axes[2, 1].set_xticks(fold_nums)
axes[2, 1].grid(axis="y", alpha=0.3)
axes[2, 1].legend()

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RESULTS_FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
log.info(f"\nResults figure saved as {RESULTS_FIGURE_PATH}")

# =========================
# Save final model
# =========================

torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
log.info(f"Model saved as {MODEL_SAVE_PATH}")
