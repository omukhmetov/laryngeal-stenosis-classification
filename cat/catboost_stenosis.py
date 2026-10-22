"""
CatBoost Stenosis Grade Classifier
===================================
Improvements over v1:
  - Drops rows where target is 'no data'
  - Cleans mixed-type columns (Age, re-intubation, comorbidity strings)
  - Computes imputation medians on TRAIN only (no leakage)
  - Ordinal label encoding for the target (1→0, 2→1, 3→2, 4→3)
  - Group-aware cross-validation for stable estimates
  - Final hold-out evaluation on a true test set
  - SHAP-style feature importance (ShapValues via CatBoost)
  - Optuna hyperparameter search (optional, set RUN_OPTUNA = True)
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import warnings
import os
import tempfile
from itertools import combinations, product
import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib"),
)


try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - for older scikit-learn installs
    StratifiedGroupKFold = None

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH = "DATA.xlsx"
RANDOM_SEED = 42
TEST_SIZE = 0.20
N_CV_FOLDS = 5
N_CV_REPEATS = 5
RUN_OPTUNA = False   # set True for hyperparameter search (requires optuna)
RUN_MODEL_COMPARISON = True
SELECTION_METRIC = "qwk"  # use "accuracy" or "qwk"
PREDICTION_TUNING_METRIC = "accuracy"  # use "accuracy" or "qwk"
RUN_PREDICTION_TUNING = True
ALLOW_TUNING_QWK_DROP = 0.01
OOF_TUNING_REPEATS = 5
OPTUNA_TRIALS = 40
EARLY_STOPPING_ROUNDS = 50
GROUP_COL = "Patient №№"
ID_COLUMNS = ["№№", "Patient №№", "Year"]
MISSING_TOKENS = {"", "no data", "not reported", "nan", "none"}
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# 1. Load
# =============================================================================
df = pd.read_excel(DATA_PATH, header=1)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

target_col = "Grade of stenosis"


# =============================================================================
# 2. Clean
# =============================================================================

# 2a. Drop rows where the target is unknown
df = df[df[target_col].astype(str).str.strip().str.lower() != "no data"].copy()
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)
print(f"After dropping 'no data' targets: {df.shape[0]} rows")


def make_groups(frame, group_col):
    """Build patient groups for leakage-safe splitting."""
    if group_col not in frame.columns:
        fallback = np.array([f"row_{i}" for i in range(len(frame))])
        print(
            f"\nGrouping column '{group_col}' not found; using row-level splits.")
        return fallback, False

    group_values = frame[group_col].astype("string").str.strip()
    missing_group = group_values.isna() | group_values.str.lower().isin(MISSING_TOKENS)
    row_fallback = pd.Series(
        [f"row_{idx}" for idx in frame.index],
        index=frame.index,
        dtype="string",
    )
    group_values = group_values.mask(missing_group, row_fallback)
    groups = group_values.astype(str).to_numpy()
    duplicated_rows = int(pd.Series(groups).duplicated(keep=False).sum())
    print(
        f"\nPatient groups: {len(np.unique(groups))} unique; "
        f"{duplicated_rows} rows share a patient/group ID."
    )
    return groups, True


groups, using_patient_groups = make_groups(df, GROUP_COL)

# 2b. Age at primary intubation — treat 'up to 1' as 0.5, 'not reported' as NaN


def clean_age(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s == "up to 1":
        return 0.5
    if s in ("not reported", "no data", ""):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


age_col = "Age at primary intubation (months)"
if age_col in df.columns:
    df[age_col] = df[age_col].apply(clean_age)

# 2c. re-intubation — 'no data' → NaN, then numeric
if "re-intubation" in df.columns:
    df["re-intubation"] = pd.to_numeric(
        df["re-intubation"].astype(str).str.strip().str.lower().replace(
            list(MISSING_TOKENS), np.nan
        ),
        errors="coerce",
    )

# 2d. Numeric columns stored as object
numeric_cols = [
    "mother's pregnancy №",
    "childbirth №",
    "Birth weight, gramm",
    "Height at birth, cm",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 2e. Strip trailing spaces from string columns and normalize missing-like tokens
str_cols = df.select_dtypes(include=["object", "str"]).columns
for col in str_cols:
    cleaned = df[col].astype("string").str.strip()
    df[col] = cleaned.mask(cleaned.str.lower().isin(MISSING_TOKENS), pd.NA)


# =============================================================================
# 3. Features / target
# =============================================================================
y_raw = df[target_col].values  # integer grades 1–4
X = df.drop(columns=[target_col] + [c for c in ID_COLUMNS if c in df.columns])

# Encode target as 0-based integers for CatBoost MultiClass
le = LabelEncoder()
y = le.fit_transform(y_raw)          # 1→0, 2→1, 3→2, 4→3
class_names = [str(c) for c in le.classes_]
print(f"\nTarget classes: {le.classes_}  →  encoded: {np.unique(y)}")
class_distribution = {
    int(cls): int(count)
    for cls, count in zip(le.classes_, np.bincount(y))
}
print(f"Class distribution: {class_distribution}")

# Identify categorical columns — compatible with pandas 2.x (object) and 3.x (string)


def is_string_col(series):
    return (
        series.dtype == object
        or str(series.dtype) == "string"
        or (
            pd.api.types.is_string_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        )
    )


cat_features = [col for col in X.columns if is_string_col(X[col])]
num_features = [col for col in X.columns if col not in cat_features]

print(f"\nCategorical features ({len(cat_features)}): {cat_features}")
print(f"Numeric features    ({len(num_features)}): {num_features}")


# =============================================================================
# 4. Train / test split  (stratified, held out until final evaluation)
# =============================================================================
def safe_n_splits(y_values, requested):
    class_counts = np.bincount(y_values)
    present_class_counts = class_counts[class_counts > 0]
    if len(present_class_counts) == 0:
        return 2
    return max(2, min(requested, int(present_class_counts.min())))


def make_holdout_indices(X_data, y_data, group_values, test_size, seed):
    """Return train/test positional indices with patient grouping when possible."""
    desired_splits = max(2, round(1 / test_size))
    n_splits = safe_n_splits(y_data, desired_splits)
    can_group = (
        using_patient_groups
        and StratifiedGroupKFold is not None
        and len(np.unique(group_values)) >= n_splits
    )

    if can_group:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return next(splitter.split(X_data, y_data, group_values)), "StratifiedGroupKFold"

    indices = np.arange(len(y_data))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y_data,
    )
    return (train_idx, test_idx), "Stratified train_test_split"


def cv_splits(X_data, y_data, group_values, n_splits, seed):
    n_splits = safe_n_splits(y_data, n_splits)
    can_group = (
        using_patient_groups
        and StratifiedGroupKFold is not None
        and len(np.unique(group_values)) >= n_splits
    )

    if can_group:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return splitter.split(X_data, y_data, group_values), "StratifiedGroupKFold"

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return splitter.split(X_data, y_data), "StratifiedKFold"


(train_idx, test_idx), split_name = make_holdout_indices(
    X, y, groups, TEST_SIZE, RANDOM_SEED
)
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y[train_idx], y[test_idx]
groups_train_full, groups_test = groups[train_idx], groups[test_idx]

group_overlap = set(groups_train_full).intersection(set(groups_test))
print(f"\nTrain: {len(X_train_full)}, Test: {len(X_test)}")
print(f"Split method: {split_name}")
print(f"Patient/group overlap between train and test: {len(group_overlap)}")
test_distribution = {
    int(cls): int(count)
    for cls, count in zip(le.classes_, np.bincount(y_test, minlength=len(le.classes_)))
}
print(f"Test class distribution: {test_distribution}")


# =============================================================================
# 5. Preprocessing helpers  (fit on train, apply to test — no leakage)
# =============================================================================

def prepare(X_tr, X_te, cat_cols, num_cols):
    """Impute: median from train for numeric, 'missing' for categorical."""
    X_tr = X_tr.copy()
    X_te = X_te.copy()

    # Numeric: median from training set only (no leakage)
    if num_cols:
        # Cast to float — pandas 3.x may keep cols as ArrowDtype
        X_tr[num_cols] = X_tr[num_cols].apply(pd.to_numeric, errors="coerce")
        X_te[num_cols] = X_te[num_cols].apply(pd.to_numeric, errors="coerce")
        medians = X_tr[num_cols].median(numeric_only=True)
        X_tr[num_cols] = X_tr[num_cols].fillna(medians)
        X_te[num_cols] = X_te[num_cols].fillna(medians)

    # Categorical: fill with sentinel string
    for col in cat_cols:
        X_tr[col] = X_tr[col].fillna("missing").astype(str)
        X_te[col] = X_te[col].fillna("missing").astype(str)

    return X_tr, X_te


# =============================================================================
# 6. Model comparison helpers
# =============================================================================

base_params = dict(
    iterations=800,
    learning_rate=0.03,
    depth=5,
    l2_leaf_reg=5,
    border_count=128,
)


def make_catboost_model(params, verbose=0, seed=RANDOM_SEED):
    """Create a CatBoost model, omitting optional params set to None."""
    model_params = {
        key: value
        for key, value in params.items()
        if value is not None
    }
    return CatBoostClassifier(
        **model_params,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=seed,
        verbose=verbose,
    )


def metric_value(y_true, y_pred, metric):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "qwk":
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")
    raise ValueError(f"Unknown metric: {metric}")


def evaluate_params_cv(params, n_splits, repeats=1, print_folds=False):
    """Evaluate one parameter set with repeated patient/group-aware CV."""
    acc_scores, qwk_scores, best_iterations = [], [], []
    cv_name = None

    for repeat in range(repeats):
        seed = RANDOM_SEED + repeat * 101
        split_iter, cv_name = cv_splits(
            X_train_full,
            y_train_full,
            groups_train_full,
            n_splits,
            seed,
        )
        for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
            Xtr, Xva = prepare(
                X_train_full.iloc[tr_idx],
                X_train_full.iloc[va_idx],
                cat_features,
                num_features,
            )
            pool_tr = Pool(
                Xtr, y_train_full[tr_idx], cat_features=cat_features)
            pool_va = Pool(
                Xva, y_train_full[va_idx], cat_features=cat_features)

            model = make_catboost_model(
                params,
                verbose=0,
                seed=seed + fold,
            )
            model.fit(
                pool_tr,
                eval_set=pool_va,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                use_best_model=True,
            )

            preds = model.predict(Xva).flatten()
            acc = accuracy_score(y_train_full[va_idx], preds)
            qwk = cohen_kappa_score(
                y_train_full[va_idx],
                preds,
                weights="quadratic",
            )
            best_iteration = model.get_best_iteration()
            if best_iteration is None or best_iteration < 0:
                best_iterations.append(params["iterations"])
            else:
                best_iterations.append(best_iteration + 1)
            acc_scores.append(acc)
            qwk_scores.append(qwk)

            if print_folds:
                print(
                    f"    Repeat {repeat + 1}, Fold {fold}: "
                    f"Accuracy={acc:.4f}  QWK={qwk:.4f}"
                )

    return {
        "cv_method": cv_name,
        "cv_repeats": repeats,
        "mean_accuracy": float(np.mean(acc_scores)),
        "std_accuracy": float(np.std(acc_scores)),
        "mean_qwk": float(np.mean(qwk_scores)),
        "std_qwk": float(np.std(qwk_scores)),
        "median_best_iteration": int(np.median(best_iterations)),
    }


candidate_configs = [
    (
        "balanced_baseline",
        dict(base_params, auto_class_weights="Balanced"),
    ),
    (
        "sqrt_balanced_baseline",
        dict(base_params, auto_class_weights="SqrtBalanced"),
    ),
    (
        "unweighted_baseline",
        dict(base_params, auto_class_weights=None),
    ),
    (
        "balanced_shallow_regularized",
        dict(
            base_params,
            learning_rate=0.025,
            depth=3,
            l2_leaf_reg=10,
            random_strength=1.5,
            auto_class_weights="Balanced",
        ),
    ),
    (
        "unweighted_shallow_regularized",
        dict(
            base_params,
            learning_rate=0.025,
            depth=3,
            l2_leaf_reg=10,
            random_strength=1.5,
            auto_class_weights=None,
        ),
    ),
    (
        "balanced_deeper",
        dict(
            base_params,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=8,
            random_strength=1.0,
            auto_class_weights="Balanced",
        ),
    ),
    (
        "unweighted_deeper",
        dict(
            base_params,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=8,
            random_strength=1.0,
            auto_class_weights=None,
        ),
    ),
]


def predict_with_class_weights(proba, class_weights):
    return np.argmax(proba * np.asarray(class_weights), axis=1)


def predict_with_ordinal_thresholds(proba, thresholds):
    expected_class = proba @ np.arange(proba.shape[1])
    return np.digitize(expected_class, thresholds)


def prediction_rule_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def tune_prediction_rule(y_true, proba, metric):
    """Tune post-processing on validation predictions only."""
    n_classes = proba.shape[1]
    rows = []

    def add_candidate(rule, preds, class_weights=None, thresholds=None):
        scores = prediction_rule_metrics(y_true, preds)
        rows.append({
            "Rule": rule,
            "accuracy": scores["accuracy"],
            "qwk": scores["qwk"],
            "class_weights": (
                None
                if class_weights is None
                else ",".join(f"{w:.2f}" for w in class_weights)
            ),
            "thresholds": (
                None
                if thresholds is None
                else ",".join(f"{t:.2f}" for t in thresholds)
            ),
            "_preds": preds,
            "_weights": class_weights,
            "_thresholds": thresholds,
        })

    add_candidate("argmax", np.argmax(proba, axis=1))

    weight_grid = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    for rest_weights in product(weight_grid, repeat=n_classes - 1):
        class_weights = np.array((1.0, *rest_weights), dtype=float)
        preds = predict_with_class_weights(proba, class_weights)
        add_candidate("class_weighted_argmax", preds,
                      class_weights=class_weights)

    threshold_grid = np.arange(0.25, n_classes - 1, 0.25)
    for thresholds in combinations(threshold_grid, n_classes - 1):
        thresholds = np.array(thresholds, dtype=float)
        preds = predict_with_ordinal_thresholds(proba, thresholds)
        add_candidate("ordinal_thresholds", preds, thresholds=thresholds)

    tuning_df = pd.DataFrame(rows)
    tie_breaker = "qwk" if metric == "accuracy" else "accuracy"
    best_index = max(
        tuning_df.index,
        key=lambda idx: (
            tuning_df.at[idx, metric],
            tuning_df.at[idx, tie_breaker],
        ),
    )
    best = tuning_df.loc[best_index].to_dict()
    clean_df = (
        tuning_df
        .drop(columns=["_preds", "_weights", "_thresholds"])
        .sort_values([metric, tie_breaker], ascending=False)
        .reset_index(drop=True)
    )
    return best, clean_df


def apply_prediction_rule(proba, rule_info):
    rule = rule_info["Rule"]
    if rule == "argmax":
        return np.argmax(proba, axis=1)
    if rule == "class_weighted_argmax":
        return predict_with_class_weights(proba, rule_info["_weights"])
    if rule == "ordinal_thresholds":
        return predict_with_ordinal_thresholds(proba, rule_info["_thresholds"])
    raise ValueError(f"Unknown prediction rule: {rule}")


def collect_oof_proba(params, n_splits, repeats):
    """Collect repeated out-of-fold probabilities for prediction-rule tuning."""
    n_classes = len(class_names)
    proba_sum = np.zeros((len(X_train_full), n_classes), dtype=float)
    proba_counts = np.zeros(len(X_train_full), dtype=float)
    cv_name = None

    for repeat in range(repeats):
        seed = RANDOM_SEED + 1009 + repeat * 101
        split_iter, cv_name = cv_splits(
            X_train_full,
            y_train_full,
            groups_train_full,
            n_splits,
            seed,
        )
        for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
            Xtr, Xva = prepare(
                X_train_full.iloc[tr_idx],
                X_train_full.iloc[va_idx],
                cat_features,
                num_features,
            )
            pool_tr = Pool(
                Xtr, y_train_full[tr_idx], cat_features=cat_features)
            pool_va = Pool(
                Xva, y_train_full[va_idx], cat_features=cat_features)
            model = make_catboost_model(
                params,
                verbose=0,
                seed=seed + fold,
            )
            model.fit(
                pool_tr,
                eval_set=pool_va,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                use_best_model=True,
            )
            proba_sum[va_idx] += model.predict_proba(Xva)
            proba_counts[va_idx] += 1

    if np.any(proba_counts == 0):
        raise RuntimeError(
            "Some training rows did not receive OOF predictions.")

    return proba_sum / proba_counts[:, None], cv_name


# =============================================================================
# 7. (Optional) Optuna hyperparameter search
# =============================================================================

if RUN_OPTUNA:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        class_weight_choice = trial.suggest_categorical(
            "auto_class_weights",
            ["None", "Balanced", "SqrtBalanced"],
        )
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
            "random_strength": trial.suggest_float("random_strength", 0.1, 3.0),
            "auto_class_weights": (
                None if class_weight_choice == "None" else class_weight_choice
            ),
        }
        results = evaluate_params_cv(
            params,
            n_splits=3,
            repeats=1,
            print_folds=False,
        )
        metric_key = "mean_accuracy" if SELECTION_METRIC == "accuracy" else "mean_qwk"
        return results[metric_key]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    optuna_params = dict(study.best_params)
    if optuna_params["auto_class_weights"] == "None":
        optuna_params["auto_class_weights"] = None
    candidate_configs.append(("optuna_best", optuna_params))
    print(f"\nOptuna best {SELECTION_METRIC}: {study.best_value:.4f}")
    print(f"Optuna params: {optuna_params}")


# =============================================================================
# 8. Cross-validation model comparison on the training set
# =============================================================================
print(f"\n{'='*60}")
print(f"  {N_CV_REPEATS}×{N_CV_FOLDS}-Fold Model Comparison")
print(f"{'='*60}")

comparison_rows = []
if RUN_MODEL_COMPARISON:
    for name, params in candidate_configs:
        results = evaluate_params_cv(
            params,
            n_splits=N_CV_FOLDS,
            repeats=N_CV_REPEATS,
            print_folds=False,
        )
        row = {"Config": name, **results, "params": params}
        comparison_rows.append(row)
        print(
            f"  {name:<31} "
            f"Accuracy={results['mean_accuracy']:.4f}±{results['std_accuracy']:.4f}  "
            f"QWK={results['mean_qwk']:.4f}±{results['std_qwk']:.4f}  "
            f"median_iter={results['median_best_iteration']}"
        )
else:
    params = dict(base_params, auto_class_weights="Balanced")
    results = evaluate_params_cv(
        params,
        n_splits=N_CV_FOLDS,
        repeats=N_CV_REPEATS,
        print_folds=True,
    )
    comparison_rows.append({
        "Config": "balanced_baseline",
        **results,
        "params": params,
    })

metric_key = "mean_accuracy" if SELECTION_METRIC == "accuracy" else "mean_qwk"
tie_breaker = "mean_qwk" if SELECTION_METRIC == "accuracy" else "mean_accuracy"
best_row = max(
    comparison_rows,
    key=lambda row: (row[metric_key], row[tie_breaker], -row["std_accuracy"]),
)
best_params = dict(best_row["params"])

comparison_df = (
    pd.DataFrame(comparison_rows)
    .drop(columns=["params"])
    .sort_values([metric_key, tie_breaker], ascending=False)
    .reset_index(drop=True)
)

print("\nModel Comparison Summary:")
print(
    comparison_df[
        [
            "Config",
            "cv_repeats",
            "mean_accuracy",
            "std_accuracy",
            "mean_qwk",
            "std_qwk",
            "median_best_iteration",
        ]
    ].to_string(index=False)
)
print(f"\nSelected config: {best_row['Config']}")
print(f"Selection metric: {SELECTION_METRIC}")
print(f"Selected params: {best_params}")

if RUN_PREDICTION_TUNING:
    print(
        f"\nTuning prediction rule with "
        f"{OOF_TUNING_REPEATS}×{N_CV_FOLDS}-fold out-of-fold probabilities..."
    )
    oof_proba, oof_cv_name = collect_oof_proba(
        best_params,
        n_splits=N_CV_FOLDS,
        repeats=OOF_TUNING_REPEATS,
    )
    raw_oof_pred = np.argmax(oof_proba, axis=1)
    raw_oof_scores = prediction_rule_metrics(y_train_full, raw_oof_pred)
    candidate_prediction_rule, prediction_tuning_df = tune_prediction_rule(
        y_train_full,
        oof_proba,
        PREDICTION_TUNING_METRIC,
    )
    raw_prediction_rule = {
        "Rule": "argmax",
        "accuracy": raw_oof_scores["accuracy"],
        "qwk": raw_oof_scores["qwk"],
        "class_weights": None,
        "thresholds": None,
        "_weights": None,
        "_thresholds": None,
    }
    tuning_improves_metric = (
        candidate_prediction_rule[PREDICTION_TUNING_METRIC]
        > raw_oof_scores[PREDICTION_TUNING_METRIC]
    )
    tuning_preserves_qwk = (
        candidate_prediction_rule["qwk"]
        >= raw_oof_scores["qwk"] - ALLOW_TUNING_QWK_DROP
    )
    if tuning_improves_metric and tuning_preserves_qwk:
        prediction_rule = candidate_prediction_rule
        tuning_decision = "applied"
    else:
        prediction_rule = raw_prediction_rule
        tuning_decision = "rejected; using raw argmax"
else:
    oof_cv_name = "not run"
    raw_oof_scores = {"accuracy": np.nan, "qwk": np.nan}
    candidate_prediction_rule = None
    prediction_rule = {
        "Rule": "argmax",
        "accuracy": np.nan,
        "qwk": np.nan,
        "class_weights": None,
        "thresholds": None,
        "_weights": None,
        "_thresholds": None,
    }
    prediction_tuning_df = pd.DataFrame([{
        "Rule": "argmax",
        "accuracy": np.nan,
        "qwk": np.nan,
        "class_weights": None,
        "thresholds": None,
    }])
    tuning_decision = "not run"

print(
    "OOF raw argmax: "
    f"Accuracy={raw_oof_scores['accuracy']:.4f}  "
    f"QWK={raw_oof_scores['qwk']:.4f}"
)
print(
    "Best tuned OOF candidate: "
    f"{candidate_prediction_rule['Rule']} "
    f"(Accuracy={candidate_prediction_rule['accuracy']:.4f}, "
    f"QWK={candidate_prediction_rule['qwk']:.4f})"
    if candidate_prediction_rule is not None
    else "Best tuned OOF candidate: not run"
)
print(f"Prediction tuning decision: {tuning_decision}")
print(
    "Selected prediction rule: "
    f"{prediction_rule['Rule']} "
    f"(OOF Accuracy={prediction_rule['accuracy']:.4f}, "
    f"QWK={prediction_rule['qwk']:.4f})"
)
if isinstance(prediction_rule["class_weights"], str):
    print(f"Class probability weights: {prediction_rule['class_weights']}")
if isinstance(prediction_rule["thresholds"], str):
    print(f"Ordinal thresholds: {prediction_rule['thresholds']}")


# =============================================================================
# 9. Final model — trained on all training data, evaluated on held-out test
# =============================================================================
print(f"\n{'='*60}")
print("  Final Model — Hold-Out Test Evaluation")
print(f"{'='*60}")

(fit_idx, val_idx), val_split_name = make_holdout_indices(
    X_train_full, y_train_full, groups_train_full, TEST_SIZE, RANDOM_SEED + 1
)

X_fit, X_val = X_train_full.iloc[fit_idx], X_train_full.iloc[val_idx]
y_fit, y_val = y_train_full[fit_idx], y_train_full[val_idx]

X_fit_prep, X_val_prep = prepare(X_fit, X_val, cat_features, num_features)
fit_pool = Pool(X_fit_prep, y_fit, cat_features=cat_features)
val_pool = Pool(X_val_prep, y_val, cat_features=cat_features)

print(f"Validation split method: {val_split_name}")
print(f"Fit: {len(X_fit)}, Validation: {len(X_val)}, Test: {len(X_test)}")

tuning_model = make_catboost_model(best_params, verbose=100)
tuning_model.fit(
    fit_pool,
    eval_set=val_pool,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    use_best_model=True,
)

val_proba = tuning_model.predict_proba(X_val_prep)
raw_val_pred = np.argmax(val_proba, axis=1)
rule_val_pred = apply_prediction_rule(val_proba, prediction_rule)
raw_val_scores = prediction_rule_metrics(y_val, raw_val_pred)
rule_val_scores = prediction_rule_metrics(y_val, rule_val_pred)

print(
    "\nValidation raw argmax: "
    f"Accuracy={raw_val_scores['accuracy']:.4f}  "
    f"QWK={raw_val_scores['qwk']:.4f}"
)
print(
    "OOF-selected rule on validation: "
    f"Accuracy={rule_val_scores['accuracy']:.4f}  "
    f"QWK={rule_val_scores['qwk']:.4f}"
)

best_iteration = tuning_model.get_best_iteration()
if best_iteration is None or best_iteration < 0:
    final_iterations = best_params["iterations"]
else:
    final_iterations = max(1, best_iteration + 1)

final_params = dict(best_params)
final_params["iterations"] = final_iterations
print(
    f"Best validation iteration: {best_iteration}; "
    f"final iterations: {final_iterations}"
)

X_train_prep, X_test_prep = prepare(
    X_train_full, X_test, cat_features, num_features
)
train_pool = Pool(X_train_prep, y_train_full, cat_features=cat_features)

final_model = make_catboost_model(final_params, verbose=100)
final_model.fit(train_pool)
y_pred_proba = final_model.predict_proba(X_test_prep)
y_pred_raw = np.argmax(y_pred_proba, axis=1)
y_pred = apply_prediction_rule(y_pred_proba, prediction_rule)

majority_class = np.bincount(y_train_full).argmax()
baseline_pred = np.full_like(y_test, majority_class)
print(
    f"\nMajority Baseline Accuracy: {accuracy_score(y_test, baseline_pred):.4f}"
)
print(
    f"Majority Baseline QWK     : "
    f"{cohen_kappa_score(y_test, baseline_pred, weights='quadratic'):.4f}"
)
print(f"\nRaw Argmax Test Accuracy : {accuracy_score(y_test, y_pred_raw):.4f}")
print(
    f"Raw Argmax Test QWK      : "
    f"{cohen_kappa_score(y_test, y_pred_raw, weights='quadratic'):.4f}"
)
print(f"\nSelected Rule Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(
    f"Selected Rule Test QWK     : "
    f"{cohen_kappa_score(y_test, y_pred, weights='quadratic'):.4f}"
)

all_labels = np.arange(len(class_names))
report_dict = classification_report(
    y_test,
    y_pred,
    labels=all_labels,
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)
report_df = pd.DataFrame(report_dict).transpose()

print("\nClassification Report (grades 1–4):")
print(
    classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=class_names,
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, y_pred, labels=all_labels)
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=[f"True {c}" for c in class_names],
                   columns=[f"Pred {c}" for c in class_names]))


# =============================================================================
# 10. Feature importance (PredictionValuesChange + SHAP)
# =============================================================================

# Standard importance
importance_df = pd.DataFrame({
    "Feature":    X_train_prep.columns,
    "Importance": final_model.get_feature_importance(train_pool),
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\nFeature Importance (top 15):")
print(importance_df.head(15).to_string(index=False))


def mean_abs_shap(shap_values, n_features):
    """Collapse CatBoost SHAP values to one mean absolute value per feature."""
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3 and shap_array.shape[-1] == n_features + 1:
        feature_values = shap_array[..., :n_features]
        return np.abs(feature_values).mean(axis=tuple(range(feature_values.ndim - 1)))
    if shap_array.ndim == 3 and shap_array.shape[1] == n_features + 1:
        feature_values = shap_array[:, :n_features, :]
        return np.abs(feature_values).mean(axis=(0, 2))
    if shap_array.ndim == 2 and shap_array.shape[1] == n_features + 1:
        return np.abs(shap_array[:, :n_features]).mean(axis=0)
    raise ValueError(f"Unexpected SHAP array shape: {shap_array.shape}")


try:
    shap_values = final_model.get_feature_importance(
        train_pool, type="ShapValues")
    shap_importance_df = pd.DataFrame({
        "Feature": X_train_prep.columns,
        "MeanAbsSHAP": mean_abs_shap(shap_values, len(X_train_prep.columns)),
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    print("\nMean Absolute SHAP Importance (top 15):")
    print(shap_importance_df.head(15).to_string(index=False))
except Exception as exc:
    shap_importance_df = pd.DataFrame(columns=["Feature", "MeanAbsSHAP"])
    print(f"\nSHAP importance skipped: {exc}")

# ── Plots ────────────────────────────────────────────────────────────────────


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)
    print(f"Plot saved: {path}")


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)


def plot_confusion(ax, matrix, title, normalize=False):
    values = matrix.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_sums,
            out=np.zeros_like(values, dtype=float),
            where=row_sums != 0,
        )
    im = ax.imshow(values, interpolation="nearest", cmap="Blues")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([f"Pred {c}" for c in class_names])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels([f"True {c}" for c in class_names])
    ax.set_title(title)
    ax.set_ylabel("True Grade")
    ax.set_xlabel("Predicted Grade")
    threshold = values.max() / 2.0 if values.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = f"{values[i, j]:.0%}" if normalize else str(matrix[i, j])
            color = "white" if values[i, j] > threshold else "black"
            ax.text(j, i, label, ha="center", va="center", color=color)
    return im


metric_columns = ["mean_accuracy", "mean_qwk"]
plot_comparison_df = comparison_df.iloc[::-1].reset_index(drop=True)
report_classes_df = report_df.loc[class_names, [
    "precision", "recall", "f1-score"]]
actual_counts = np.bincount(y_test, minlength=len(class_names))
pred_counts = np.bincount(y_pred, minlength=len(class_names))
confidence = np.max(y_pred_proba, axis=1)
is_correct = y_pred == y_test

fig, axes = plt.subplots(3, 2, figsize=(18, 18))
axes = axes.ravel()

# Model comparison
ax = axes[0]
y_pos = np.arange(len(plot_comparison_df))
bar_height = 0.36
ax.barh(
    y_pos - bar_height / 2,
    plot_comparison_df["mean_accuracy"],
    height=bar_height,
    label="Accuracy",
    color="#4C72B0",
)
ax.barh(
    y_pos + bar_height / 2,
    plot_comparison_df["mean_qwk"],
    height=bar_height,
    label="QWK",
    color="#55A868",
)
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_comparison_df["Config"])
ax.set_xlim(
    min(0, plot_comparison_df[metric_columns].min().min() - 0.05),
    min(1, plot_comparison_df[metric_columns].max().max() + 0.1),
)
ax.set_title("Model Comparison: Grouped CV")
ax.legend()
style_axis(ax)

# Standard feature importance
ax = axes[1]
top_importance = importance_df.head(12).iloc[::-1]
bars = ax.barh(
    top_importance["Feature"],
    top_importance["Importance"],
    color="#4C72B0",
)
ax.set_title("PredictionValuesChange Importance")
ax.set_xlabel("Importance")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
style_axis(ax)

# Count confusion matrix
im = plot_confusion(axes[2], cm, "Confusion Matrix: Counts", normalize=False)
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

# Normalized confusion matrix
im = plot_confusion(axes[3], cm, "Confusion Matrix: Row %", normalize=True)
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

# Actual vs predicted distribution
ax = axes[4]
x = np.arange(len(class_names))
width = 0.36
ax.bar(x - width / 2, actual_counts, width, label="Actual", color="#4C72B0")
ax.bar(x + width / 2, pred_counts, width, label="Predicted", color="#C44E52")
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_xlabel("Grade")
ax.set_ylabel("Patients")
ax.set_title("Actual vs Predicted Grade Distribution")
ax.legend()
style_axis(ax)

# Per-class metrics
ax = axes[5]
x = np.arange(len(class_names))
width = 0.25
for offset, metric, color in [
    (-width, "precision", "#4C72B0"),
    (0, "recall", "#55A868"),
    (width, "f1-score", "#C44E52"),
]:
    ax.bar(x + offset, report_classes_df[metric],
           width, label=metric, color=color)
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Grade")
ax.set_title("Per-Class Test Metrics")
ax.legend()
style_axis(ax)

save_plot(fig, "catboost_results.png")


fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# SHAP feature importance
ax = axes[0]
if not shap_importance_df.empty:
    top_shap = shap_importance_df.head(12).iloc[::-1]
    bars = ax.barh(top_shap["Feature"],
                   top_shap["MeanAbsSHAP"], color="#55A868")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
else:
    ax.text(0.5, 0.5, "SHAP unavailable", ha="center", va="center")
ax.set_title("Mean Absolute SHAP Importance")
ax.set_xlabel("Mean |SHAP|")
style_axis(ax)

# Prediction confidence
ax = axes[1]
bins = np.linspace(0, 1, 11)
ax.hist(
    confidence[is_correct],
    bins=bins,
    alpha=0.75,
    label="Correct",
    color="#55A868",
    edgecolor="white",
)
ax.hist(
    confidence[~is_correct],
    bins=bins,
    alpha=0.75,
    label="Incorrect",
    color="#C44E52",
    edgecolor="white",
)
ax.set_xlim(0, 1)
ax.set_xlabel("Predicted class probability")
ax.set_ylabel("Patients")
ax.set_title("Prediction Confidence on Test Set")
ax.legend()
style_axis(ax)

save_plot(fig, "catboost_interpretability.png")


fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# Model comparison focused figure
ax = axes[0]
plot_df = comparison_df.sort_values("mean_accuracy")
y_pos = np.arange(len(plot_df))
ax.barh(
    y_pos,
    plot_df["mean_accuracy"],
    xerr=plot_df["std_accuracy"],
    color="#4C72B0",
    alpha=0.85,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["Config"])
ax.set_xlabel("Mean CV Accuracy")
ax.set_title("Grouped CV Accuracy by Config")
style_axis(ax)

ax = axes[1]
plot_df = comparison_df.sort_values("mean_qwk")
y_pos = np.arange(len(plot_df))
ax.barh(
    y_pos,
    plot_df["mean_qwk"],
    xerr=plot_df["std_qwk"],
    color="#55A868",
    alpha=0.85,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["Config"])
ax.set_xlabel("Mean CV QWK")
ax.set_title("Grouped CV QWK by Config")
style_axis(ax)

save_plot(fig, "catboost_model_comparison.png")

# =============================================================================
# 11. Save outputs
# =============================================================================
with pd.ExcelWriter("catboost_feature_importance.xlsx") as writer:
    comparison_df.to_excel(
        writer,
        sheet_name="ModelComparison",
        index=False,
    )
    report_df.to_excel(
        writer,
        sheet_name="ClassificationReport",
    )
    prediction_tuning_df.to_excel(
        writer,
        sheet_name="PredictionTuning",
        index=False,
    )
    importance_df.to_excel(
        writer,
        sheet_name="PredictionValuesChange",
        index=False,
    )
    shap_importance_df.to_excel(
        writer,
        sheet_name="MeanAbsSHAP",
        index=False,
    )
final_model.save_model("stenosis_catboost_model.cbm")

print("\nDone. Files saved:")
print("  catboost_feature_importance.xlsx")
print("  catboost_results.png")
print("  catboost_interpretability.png")
print("  catboost_model_comparison.png")
print("  stenosis_catboost_model.cbm")
