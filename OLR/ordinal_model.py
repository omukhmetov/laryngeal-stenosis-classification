import os
import pickle
import tempfile
import warnings
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel

project_dir = Path(__file__).resolve().parent
figures_dir = project_dir / "figures"
figures_dir.mkdir(exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(figures_dir / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - for older scikit-learn installs
    StratifiedGroupKFold = None

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 1. Configuration
# =========================

data_path = project_dir / "DATA.xlsx"
sheet_name = "All years eng"
missing_value_labels = ["no data", "not reported"]

RANDOM_SEED = 42
TEST_SIZE = 0.20
N_CV_FOLDS = 5
N_CV_REPEATS = 1
OOF_TUNING_REPEATS = 1
SELECTION_METRIC = "qwk"  # use "accuracy" or "qwk"
PREDICTION_TUNING_METRIC = "accuracy"  # use "accuracy" or "qwk"
ALLOW_TUNING_QWK_DROP = 0.01
N_PERMUTATION_REPEATS = 30

target_col = "Grade of stenosis"
age_col = "Age at primary intubation (months)"
GROUP_COL = "Patient №№"
ID_COLUMNS = ["№№", "Patient №№", "Year"]
MISSING_TOKENS = {"", "no data", "not reported", "nan", "none"}

numeric_features = [
    age_col,
    "re-intubation",
    "history of tracheostomy",
    "presence of T-tube",
    "mother's pregnancy №",
    "childbirth №",
    "Birth weight, gramm",
    "Height at birth, cm",
]

categorical_features = [
    "Gender",
    "Localization of stenosis",
    "Reason of intubation",
]

reference_categories = {
    "Gender": "f",
    "Localization of stenosis": "subglottic",
    "Reason of intubation": "infection",
}
minimum_category_count = 10

candidate_configs = [
    ("ordered_logit", "logit"),
    ("ordered_probit", "probit"),
]

feature_workbook_path = project_dir / "ordinal_feature_importance.xlsx"
prediction_export_path = project_dir / "ordinal_test_predictions.xlsx"
model_path = project_dir / "stenosis_ordinal_model.pkl"


# =========================
# 2. Load and clean data
# =========================

df = pd.read_excel(
    data_path,
    sheet_name=sheet_name,
    header=1,
    na_values=missing_value_labels,
)
raw_row_count = len(df)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

required_columns = numeric_features + categorical_features + [target_col]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in {data_path}: {missing_columns}")

id_columns = [col for col in ID_COLUMNS if col in df.columns]
df = df[id_columns + required_columns].copy()
df["source_excel_row"] = df.index.to_numpy() + 3

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)
target_clean_row_count = len(df)

# The spreadsheet contains "up to 1" for infants younger than 1 month.
# This uses the midpoint convention; change to 1.0 if your protocol requires it.
df[age_col] = df[age_col].replace({"up to 1": 0.5})

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Localization of stenosis"] = df["Localization of stenosis"].replace({
    "надglottic": "supraglottic",
    "refusal": pd.NA,
})

for col in categorical_features:
    cleaned = df[col].astype("string").str.strip()
    df[col] = cleaned.mask(cleaned.str.lower().isin(MISSING_TOKENS), pd.NA)

rare_category_groups = {}
for col in categorical_features:
    reference = reference_categories[col]
    category_counts = df[col].value_counts(dropna=True)
    rare_categories = [
        category
        for category, count in category_counts.items()
        if count < minimum_category_count and category != reference
    ]

    if rare_categories:
        df.loc[df[col].isin(rare_categories), col] = "Other"
        rare_category_groups[col] = [str(category) for category in rare_categories]

for col in categorical_features:
    reference = reference_categories[col]
    categories = [reference] + sorted(
        value for value in df[col].dropna().unique() if value != reference
    )
    df[col] = pd.Categorical(df[col], categories=categories)

df = df.dropna(subset=numeric_features + categorical_features)
model_row_count = len(df)


# =========================
# 3. Features, target, groups
# =========================

X_numeric = df[numeric_features].astype(float)
X_categorical = pd.get_dummies(
    df[categorical_features],
    drop_first=True,
    dtype=float,
)
X = pd.concat([X_numeric, X_categorical], axis=1).astype(float)
predictor_columns = X.columns.tolist()
y = df[target_col].astype(int)
class_labels = np.array(sorted(y.unique()), dtype=int)
class_names = [str(label) for label in class_labels]


def make_groups(frame, group_col):
    """Build patient groups for leakage-aware splitting."""
    if group_col not in frame.columns:
        fallback = np.array([f"row_{idx}" for idx in frame.index])
        print(f"\nGrouping column '{group_col}' not found; using row-level splits.")
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

print(f"\nTarget classes: {class_labels}")
class_distribution = y.value_counts().sort_index().astype(int).to_dict()
print(f"Class distribution: {class_distribution}")
print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")
print(f"Encoded predictor columns ({len(predictor_columns)}): {predictor_columns}")


# =========================
# 4. Splitting helpers
# =========================

def safe_n_splits(y_values, requested):
    _, counts = np.unique(y_values, return_counts=True)
    if len(counts) == 0:
        return 2
    return max(2, min(requested, int(counts.min())))


def can_use_group_split(group_values, n_splits):
    return (
        using_patient_groups
        and StratifiedGroupKFold is not None
        and len(np.unique(group_values)) >= n_splits
    )


def make_holdout_indices(X_data, y_values, group_values, test_size, seed):
    desired_splits = max(2, round(1 / test_size))
    n_splits = safe_n_splits(y_values, desired_splits)

    if can_use_group_split(group_values, n_splits):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return next(splitter.split(X_data, y_values, group_values)), "StratifiedGroupKFold"

    indices = np.arange(len(y_values))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y_values,
    )
    return (train_idx, test_idx), "Stratified train_test_split"


def cv_splits(X_data, y_values, group_values, n_splits, seed):
    n_splits = safe_n_splits(y_values, n_splits)

    if can_use_group_split(group_values, n_splits):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return splitter.split(X_data, y_values, group_values), "StratifiedGroupKFold"

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return splitter.split(X_data, y_values), "StratifiedKFold"


(train_idx, test_idx), split_name = make_holdout_indices(
    X, y.to_numpy(), groups, TEST_SIZE, RANDOM_SEED
)
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train_full, groups_test = groups[train_idx], groups[test_idx]

group_overlap = set(groups_train_full).intersection(set(groups_test))
print(f"\nTrain: {len(X_train_full)}, Test: {len(X_test)}")
print(f"Split method: {split_name}")
print(f"Patient/group overlap between train and test: {len(group_overlap)}")
test_distribution = y_test.value_counts().sort_index().astype(int).to_dict()
print(f"Test class distribution: {test_distribution}")


# =========================
# 5. Model and prediction helpers
# =========================

def non_constant_columns(X_train):
    return X_train.columns[X_train.nunique(dropna=False) > 1].tolist()


def fit_ordinal_model(X_train, y_train, distr):
    fit_columns = non_constant_columns(X_train)
    model = OrderedModel(
        y_train.astype(int),
        X_train[fit_columns].astype(float),
        distr=distr,
    )
    result = model.fit(method="bfgs", disp=False, maxiter=1000)
    return result, fit_columns


def predict_proba_frame(result, X_data, fit_columns):
    proba = np.asarray(
        result.model.predict(result.params, exog=X_data[fit_columns].astype(float))
    )
    if proba.ndim != 2 or proba.shape[1] != len(class_labels):
        raise ValueError(f"Unexpected probability shape: {proba.shape}")
    return pd.DataFrame(
        proba,
        columns=class_labels,
        index=X_data.index,
    )


def raw_argmax_prediction(proba):
    return class_labels[np.argmax(np.asarray(proba), axis=1)]


def predict_with_class_weights(proba, class_weights):
    weighted_index = np.argmax(np.asarray(proba) * np.asarray(class_weights), axis=1)
    return class_labels[weighted_index]


def predict_with_ordinal_thresholds(proba, thresholds):
    expected_grade = np.asarray(proba) @ class_labels
    predicted_index = np.digitize(expected_grade, thresholds)
    return class_labels[predicted_index]


def prediction_rule_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def tune_prediction_rule(y_true, proba, metric):
    """Tune post-processing on validation predictions only."""
    n_classes = len(class_labels)
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
            "_weights": class_weights,
            "_thresholds": thresholds,
        })

    add_candidate("argmax", raw_argmax_prediction(proba))

    weight_grid = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    for rest_weights in product(weight_grid, repeat=n_classes - 1):
        class_weights = np.array((1.0, *rest_weights), dtype=float)
        preds = predict_with_class_weights(proba, class_weights)
        add_candidate("class_weighted_argmax", preds, class_weights=class_weights)

    threshold_grid = np.arange(class_labels[0] + 0.25, class_labels[-1], 0.25)
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
        .drop(columns=["_weights", "_thresholds"])
        .sort_values([metric, tie_breaker], ascending=False)
        .reset_index(drop=True)
    )
    return best, clean_df


def apply_prediction_rule(proba, rule_info):
    rule = rule_info["Rule"]
    if rule == "argmax":
        return raw_argmax_prediction(proba)
    if rule == "class_weighted_argmax":
        return predict_with_class_weights(proba, rule_info["_weights"])
    if rule == "ordinal_thresholds":
        return predict_with_ordinal_thresholds(proba, rule_info["_thresholds"])
    raise ValueError(f"Unknown prediction rule: {rule}")


def collect_oof_proba(distr, n_splits, repeats):
    proba_sum = np.zeros((len(X_train_full), len(class_labels)), dtype=float)
    proba_counts = np.zeros(len(X_train_full), dtype=float)
    cv_name = None

    for repeat in range(repeats):
        seed = RANDOM_SEED + 1009 + repeat * 101
        split_iter, cv_name = cv_splits(
            X_train_full,
            y_train_full.to_numpy(),
            groups_train_full,
            n_splits,
            seed,
        )
        for tr_idx, va_idx in split_iter:
            result, fit_columns = fit_ordinal_model(
                X_train_full.iloc[tr_idx],
                y_train_full.iloc[tr_idx],
                distr,
            )
            proba = predict_proba_frame(
                result,
                X_train_full.iloc[va_idx],
                fit_columns,
            ).to_numpy()
            proba_sum[va_idx] += proba
            proba_counts[va_idx] += 1

    if np.any(proba_counts == 0):
        raise RuntimeError("Some training rows did not receive OOF predictions.")

    return proba_sum / proba_counts[:, None], cv_name


# =========================
# 6. Model comparison
# =========================

print(f"\n{'=' * 60}")
print(f"  {N_CV_REPEATS}x{N_CV_FOLDS}-Fold Ordinal Model Comparison")
print(f"{'=' * 60}")


def evaluate_config_cv(config_name, distr):
    acc_scores, qwk_scores = [], []
    cv_name = None

    for repeat in range(N_CV_REPEATS):
        seed = RANDOM_SEED + repeat * 101
        split_iter, cv_name = cv_splits(
            X_train_full,
            y_train_full.to_numpy(),
            groups_train_full,
            N_CV_FOLDS,
            seed,
        )
        for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
            result, fit_columns = fit_ordinal_model(
                X_train_full.iloc[tr_idx],
                y_train_full.iloc[tr_idx],
                distr,
            )
            va_proba = predict_proba_frame(
                result,
                X_train_full.iloc[va_idx],
                fit_columns,
            )
            va_pred = raw_argmax_prediction(va_proba)
            acc = accuracy_score(y_train_full.iloc[va_idx], va_pred)
            qwk = cohen_kappa_score(
                y_train_full.iloc[va_idx],
                va_pred,
                weights="quadratic",
            )
            acc_scores.append(acc)
            qwk_scores.append(qwk)
            print(
                f"    {config_name}, repeat {repeat + 1}, fold {fold}: "
                f"Accuracy={acc:.4f}  QWK={qwk:.4f}"
            )

    return {
        "Config": config_name,
        "distribution": distr,
        "cv_method": cv_name,
        "cv_repeats": N_CV_REPEATS,
        "mean_accuracy": float(np.mean(acc_scores)),
        "std_accuracy": float(np.std(acc_scores)),
        "mean_qwk": float(np.mean(qwk_scores)),
        "std_qwk": float(np.std(qwk_scores)),
    }


comparison_rows = []
for config_name, distr in candidate_configs:
    row = evaluate_config_cv(config_name, distr)
    comparison_rows.append(row)
    print(
        f"  {config_name:<18} "
        f"Accuracy={row['mean_accuracy']:.4f}+/-{row['std_accuracy']:.4f}  "
        f"QWK={row['mean_qwk']:.4f}+/-{row['std_qwk']:.4f}"
    )

metric_key = "mean_accuracy" if SELECTION_METRIC == "accuracy" else "mean_qwk"
tie_breaker = "mean_qwk" if SELECTION_METRIC == "accuracy" else "mean_accuracy"
best_row = max(
    comparison_rows,
    key=lambda row: (row[metric_key], row[tie_breaker], -row["std_accuracy"]),
)
best_distribution = best_row["distribution"]

comparison_df = (
    pd.DataFrame(comparison_rows)
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
        ]
    ].to_string(index=False)
)
print(f"\nSelected config: {best_row['Config']}")
print(f"Selection metric: {SELECTION_METRIC}")

print(
    f"\nTuning prediction rule with "
    f"{OOF_TUNING_REPEATS}x{N_CV_FOLDS}-fold out-of-fold probabilities..."
)
oof_proba, oof_cv_name = collect_oof_proba(
    best_distribution,
    n_splits=N_CV_FOLDS,
    repeats=OOF_TUNING_REPEATS,
)
raw_oof_pred = raw_argmax_prediction(oof_proba)
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


# =========================
# 7. Final hold-out evaluation
# =========================

print(f"\n{'=' * 60}")
print("  Final Ordinal Model - Hold-Out Test Evaluation")
print(f"{'=' * 60}")

result, fit_columns = fit_ordinal_model(
    X_train_full,
    y_train_full,
    best_distribution,
)
print(result.summary())

y_pred_proba_df = predict_proba_frame(result, X_test, fit_columns)
y_pred_proba = y_pred_proba_df.to_numpy()
y_pred_raw = raw_argmax_prediction(y_pred_proba)
y_pred = apply_prediction_rule(y_pred_proba, prediction_rule)

majority_class = y_train_full.value_counts().idxmax()
baseline_pred = np.full(len(y_test), majority_class, dtype=int)

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_qwk = cohen_kappa_score(y_test, baseline_pred, weights="quadratic")
raw_accuracy = accuracy_score(y_test, y_pred_raw)
raw_qwk = cohen_kappa_score(y_test, y_pred_raw, weights="quadratic")
selected_accuracy = accuracy_score(y_test, y_pred)
selected_qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

print(f"\nMajority Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Majority Baseline QWK     : {baseline_qwk:.4f}")
print(f"\nRaw Argmax Test Accuracy : {raw_accuracy:.4f}")
print(f"Raw Argmax Test QWK      : {raw_qwk:.4f}")
print(f"\nSelected Rule Test Accuracy: {selected_accuracy:.4f}")
print(f"Selected Rule Test QWK     : {selected_qwk:.4f}")

report_dict = classification_report(
    y_test,
    y_pred,
    labels=class_labels,
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)
report_df = pd.DataFrame(report_dict).transpose()

print("\nClassification Report (grades 1-4):")
print(
    classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        target_names=class_names,
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, y_pred, labels=class_labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"True {c}" for c in class_labels],
    columns=[f"Pred {c}" for c in class_labels],
)
print("Confusion Matrix:")
print(cm_df)


# =========================
# 8. Row-by-row predictions
# =========================

test_metadata = df.iloc[test_idx][["source_excel_row"] + id_columns].copy()
if GROUP_COL not in test_metadata.columns:
    test_metadata["patient_group"] = groups_test

test_predictions_df = test_metadata.reset_index(drop=True)
test_predictions_df["actual_grade"] = y_test.to_numpy()
test_predictions_df["predicted_grade"] = y_pred
test_predictions_df["predicted_probability"] = y_pred_proba[
    np.arange(len(y_pred)),
    [np.where(class_labels == pred)[0][0] for pred in y_pred],
]
test_predictions_df["raw_argmax_grade"] = y_pred_raw
test_predictions_df["raw_argmax_probability"] = y_pred_proba.max(axis=1)
test_predictions_df["selected_rule"] = prediction_rule["Rule"]
test_predictions_df["selected_rule_correct"] = y_pred == y_test.to_numpy()
test_predictions_df["raw_argmax_correct"] = y_pred_raw == y_test.to_numpy()
test_predictions_df["grade_error"] = (
    test_predictions_df["predicted_grade"]
    - test_predictions_df["actual_grade"]
)
probability_df = pd.DataFrame(
    y_pred_proba,
    columns=[f"prob_grade_{grade}" for grade in class_labels],
)
test_predictions_df = pd.concat(
    [test_predictions_df, probability_df],
    axis=1,
)
test_predictions_df.to_excel(prediction_export_path, index=False)
print(f"\nRow-by-row test predictions saved: {prediction_export_path}")


# =========================
# 9. Interpretation tables
# =========================

predictor_params = result.params.loc[fit_columns]
odds_ratios = np.exp(predictor_params)
conf_int = result.conf_int().loc[fit_columns]
or_table = pd.DataFrame({
    "Feature": fit_columns,
    "Coefficient": predictor_params.to_numpy(),
    "Odds ratio": odds_ratios.to_numpy(),
    "Lower 95% CI": np.exp(conf_int[0]).to_numpy(),
    "Upper 95% CI": np.exp(conf_int[1]).to_numpy(),
    "p_value": result.pvalues.loc[fit_columns].to_numpy(),
})
or_table["Absolute coefficient"] = or_table["Coefficient"].abs()
or_table = or_table.sort_values(
    "Absolute coefficient",
    ascending=False,
).reset_index(drop=True)

threshold_params = result.params.drop(fit_columns)
threshold_df = threshold_params.rename("Parameter").reset_index()
threshold_df.columns = ["Threshold", "Parameter"]

print("\nOdds Ratios for predictors:")
print(
    or_table[
        ["Feature", "Odds ratio", "Lower 95% CI", "Upper 95% CI", "p_value"]
    ].to_string(index=False)
)

print("\nThreshold / cutpoint parameters:")
print(threshold_df.to_string(index=False))

print("\nTop coefficient-based feature importances:")
print(
    or_table[
        ["Feature", "Coefficient", "Odds ratio", "Absolute coefficient", "p_value"]
    ].head(15).to_string(index=False)
)


def permutation_importance():
    base_metrics = prediction_rule_metrics(y_test, y_pred)
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for feature in fit_columns:
        accuracy_drops = []
        qwk_drops = []

        for _ in range(N_PERMUTATION_REPEATS):
            X_permuted = X_test.copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())
            permuted_proba = predict_proba_frame(result, X_permuted, fit_columns)
            permuted_pred = apply_prediction_rule(permuted_proba.to_numpy(), prediction_rule)
            permuted_metrics = prediction_rule_metrics(y_test, permuted_pred)
            accuracy_drops.append(base_metrics["accuracy"] - permuted_metrics["accuracy"])
            qwk_drops.append(base_metrics["qwk"] - permuted_metrics["qwk"])

        rows.append({
            "Feature": feature,
            "Mean accuracy drop": float(np.mean(accuracy_drops)),
            "Mean QWK drop": float(np.mean(qwk_drops)),
            "Std accuracy drop": float(np.std(accuracy_drops)),
            "Std QWK drop": float(np.std(qwk_drops)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["Mean QWK drop", "Mean accuracy drop"], ascending=False)
        .reset_index(drop=True)
    )


permutation_importance_df = permutation_importance()
print("\nPermutation-style feature importances (top 15):")
print(permutation_importance_df.head(15).to_string(index=False))


# =========================
# 10. Figures
# =========================

def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def style_axis(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.22)


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
    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([f"Pred {c}" for c in class_labels])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels([f"True {c}" for c in class_labels])
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


# Figure: distribution of stenosis grades in the fitted data
grade_counts = y.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(grade_counts.index.astype(str), grade_counts.values, color="#4C78A8")
ax.set_title("Distribution of Stenosis Grades")
ax.set_xlabel("Grade of stenosis")
ax.set_ylabel("Number of patients")
style_axis(ax, grid_axis="y")
save_plot(fig, figures_dir / "grade_distribution.png")

# Figure: odds ratios with 95% confidence intervals
or_plot_df = or_table.sort_values("Odds ratio")
y_pos = np.arange(len(or_plot_df))
lower_error = or_plot_df["Odds ratio"] - or_plot_df["Lower 95% CI"]
upper_error = or_plot_df["Upper 95% CI"] - or_plot_df["Odds ratio"]

fig, ax = plt.subplots(figsize=(10, max(6, 0.38 * len(or_plot_df))))
ax.errorbar(
    or_plot_df["Odds ratio"],
    y_pos,
    xerr=[lower_error, upper_error],
    fmt="o",
    color="#2F4B7C",
    ecolor="#9ECAE1",
    capsize=4,
)
ax.axvline(1, color="#D62728", linestyle="--", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(or_plot_df["Feature"])
ax.set_xscale("log")
ax.set_title("Odds Ratios With 95% Confidence Intervals")
ax.set_xlabel("Odds ratio, log scale")
style_axis(ax)
save_plot(fig, figures_dir / "odds_ratios.png")

# Figure: predicted probabilities across age, holding other predictors fixed.
if age_col in fit_columns:
    age_grid = np.linspace(X_train_full[age_col].min(), X_train_full[age_col].max(), 100)
    baseline = X_train_full[fit_columns].median()
    prediction_grid = pd.DataFrame([baseline.to_dict()] * len(age_grid))
    prediction_grid[age_col] = age_grid
    predicted_probabilities = predict_proba_frame(
        result,
        prediction_grid,
        fit_columns,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for grade in class_labels:
        ax.plot(
            age_grid,
            predicted_probabilities[grade],
            label=f"Grade {grade}",
            linewidth=2,
        )

    ax.set_title("Predicted Grade Probabilities by Age at Primary Intubation")
    ax.set_xlabel("Age at primary intubation (months)")
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0, 1)
    ax.legend(title="Outcome")
    style_axis(ax)
    save_plot(fig, figures_dir / "predicted_probabilities_by_age.png")

# Summary results figure
report_classes_df = report_df.loc[class_names, ["precision", "recall", "f1-score"]]
actual_counts = y_test.value_counts().reindex(class_labels, fill_value=0)
pred_counts = pd.Series(y_pred).value_counts().reindex(class_labels, fill_value=0)
selected_confidence = test_predictions_df["predicted_probability"].to_numpy()
is_correct = test_predictions_df["selected_rule_correct"].to_numpy()

fig, axes = plt.subplots(3, 2, figsize=(18, 18))
axes = axes.ravel()

ax = axes[0]
plot_comparison_df = comparison_df.iloc[::-1].reset_index(drop=True)
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
    min(0, plot_comparison_df[["mean_accuracy", "mean_qwk"]].min().min() - 0.05),
    min(1, plot_comparison_df[["mean_accuracy", "mean_qwk"]].max().max() + 0.10),
)
ax.set_title("Ordinal Model Comparison")
ax.legend()
style_axis(ax)

ax = axes[1]
top_coef = or_table.head(12).iloc[::-1]
bars = ax.barh(
    top_coef["Feature"],
    top_coef["Absolute coefficient"],
    color="#4C72B0",
)
ax.set_title("Coefficient-Based Importance")
ax.set_xlabel("Absolute coefficient")
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
style_axis(ax)

im = plot_confusion(axes[2], cm, "Confusion Matrix: Counts", normalize=False)
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

im = plot_confusion(axes[3], cm, "Confusion Matrix: Row %", normalize=True)
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

ax = axes[4]
x = np.arange(len(class_labels))
width = 0.36
ax.bar(x - width / 2, actual_counts, width, label="Actual", color="#4C72B0")
ax.bar(x + width / 2, pred_counts, width, label="Predicted", color="#C44E52")
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.set_xlabel("Grade")
ax.set_ylabel("Patients")
ax.set_title("Actual vs Predicted Grade Distribution")
ax.legend()
style_axis(ax)

ax = axes[5]
x = np.arange(len(class_labels))
width = 0.25
for offset, metric, color in [
    (-width, "precision", "#4C72B0"),
    (0, "recall", "#55A868"),
    (width, "f1-score", "#C44E52"),
]:
    ax.bar(
        x + offset,
        report_classes_df[metric],
        width,
        label=metric,
        color=color,
    )
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Grade")
ax.set_title("Per-Class Test Metrics")
ax.legend()
style_axis(ax)

save_plot(fig, figures_dir / "ordinal_results.png")

# Interpretability figure
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
top_perm = permutation_importance_df.head(12).iloc[::-1]
bars = ax.barh(
    top_perm["Feature"],
    top_perm["Mean QWK drop"],
    color="#55A868",
)
ax.set_title("Permutation Importance on Test Set")
ax.set_xlabel("Mean QWK drop")
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
style_axis(ax)

ax = axes[1]
bins = np.linspace(0, 1, 11)
ax.hist(
    selected_confidence[is_correct],
    bins=bins,
    alpha=0.75,
    label="Correct",
    color="#55A868",
    edgecolor="white",
)
ax.hist(
    selected_confidence[~is_correct],
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

save_plot(fig, figures_dir / "ordinal_interpretability.png")

fig, axes = plt.subplots(1, 2, figsize=(17, 7))

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
ax.set_title("CV Accuracy by Ordinal Config")
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
ax.set_title("CV QWK by Ordinal Config")
style_axis(ax)

save_plot(fig, figures_dir / "ordinal_model_comparison.png")


# =========================
# 11. Save outputs
# =========================

metrics_df = pd.DataFrame([
    {
        "Model": "Majority baseline",
        "Accuracy": baseline_accuracy,
        "QWK": baseline_qwk,
    },
    {
        "Model": "Ordinal raw argmax",
        "Accuracy": raw_accuracy,
        "QWK": raw_qwk,
    },
    {
        "Model": "Ordinal selected rule",
        "Accuracy": selected_accuracy,
        "QWK": selected_qwk,
    },
])

with pd.ExcelWriter(feature_workbook_path) as writer:
    metrics_df.to_excel(writer, sheet_name="HoldoutMetrics", index=False)
    comparison_df.to_excel(writer, sheet_name="ModelComparison", index=False)
    report_df.to_excel(writer, sheet_name="ClassificationReport")
    cm_df.to_excel(writer, sheet_name="ConfusionMatrix")
    prediction_tuning_df.to_excel(writer, sheet_name="PredictionTuning", index=False)
    or_table.to_excel(writer, sheet_name="OddsRatios", index=False)
    threshold_df.to_excel(writer, sheet_name="Thresholds", index=False)
    permutation_importance_df.to_excel(
        writer,
        sheet_name="PermutationImportance",
        index=False,
    )
    test_predictions_df.to_excel(
        writer,
        sheet_name="HoldoutTestPredictions",
        index=False,
    )

model_artifact = {
    "result": result,
    "distribution": best_distribution,
    "fit_columns": fit_columns,
    "class_labels": class_labels,
    "prediction_rule": prediction_rule,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "reference_categories": reference_categories,
}
with open(model_path, "wb") as file:
    pickle.dump(model_artifact, file)

print("\nData check:")
print(f"Rows read from Excel: {raw_row_count}")
print(f"Rows after target cleaning: {target_clean_row_count}")
print(f"Rows used in model after feature cleaning: {model_row_count}")
print("\nCategorical reference categories:")
for col, reference in reference_categories.items():
    print(f"{col}: {reference}")

if rare_category_groups:
    print(f"\nCategories with fewer than {minimum_category_count} rows grouped as Other:")
    for col, categories in rare_category_groups.items():
        print(f"{col}: {', '.join(categories)}")

print("\nDone. Files saved:")
print(f"  {feature_workbook_path}")
print(f"  {prediction_export_path}")
print(f"  {figures_dir / 'grade_distribution.png'}")
print(f"  {figures_dir / 'odds_ratios.png'}")
print(f"  {figures_dir / 'predicted_probabilities_by_age.png'}")
print(f"  {figures_dir / 'ordinal_results.png'}")
print(f"  {figures_dir / 'ordinal_interpretability.png'}")
print(f"  {figures_dir / 'ordinal_model_comparison.png'}")
print(f"  {model_path}")
