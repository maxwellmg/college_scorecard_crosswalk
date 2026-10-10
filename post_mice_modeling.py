"""
College Scorecard — post-MICE predictive modeling
====================================================
Companion to mice_pipeline.py. Loads the M completed_#.csv datasets that
script produced (0..N_DATASETS-1, in this same folder), pulls the chosen
dependent variable out of them — it rides along in every completed_#.csv
as a fully-observed, non-imputed column, so it's identical across all M
datasets by construction, no separate join needed — and fits a handful of
different model types to compare predictive accuracy.

Ensembling approach (matches the "will I be able to use all M iterations"
discussion): a model is fit separately on each of the M completed datasets,
then predictions are averaged across the M fits at test time. This is used
uniformly for every model here — including Lasso and SVM — rather than
Rubin's-rules coefficient pooling, because Rubin's rules is only well-
defined for models with a coefficient + standard error (OLS/logistic); it
doesn't have a valid extension to LASSO's shrunk/selection-unstable
coefficients or to SVM, which has no coefficient to pool at all.
Prediction-averaging works the same way regardless of model type, and as a
side effect the spread across the M individual fits' scores (reported
alongside the ensembled score below) is a rough read on how much imputation
uncertainty is contributing to the result — a model whose score barely
moves across the M fits is insensitive to which imputation you happened to
use; one that swings a lot is more imputation-sensitive.

Not implemented here (per your note that it's later work): MCMC / a
Bayesian hierarchical treatment of the multiple-imputation + prediction
problem. Everything below is a frequentist point-estimate/ensembling
approach meant for a first accuracy-and-predictive-power pass.

Dependencies: pandas, numpy, scikit-learn (`pip install scikit-learn`).
Validated end-to-end against a synthetic stand-in for mice_pipeline.py's
output (see the bottom of this file's test run in conversation) — not yet
run against your actual completed_*.csv files, since this environment
doesn't have them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

# ────────────────────────────────────────────────────────────────────────
# Config — edit these for your actual run
# ────────────────────────────────────────────────────────────────────────

COMPLETED_DIR = Path(".")   # completed_#.csv files live in this same folder
N_DATASETS = 5              # completed_0.csv .. completed_4.csv

# Every completed_#.csv carries both "Risk Score Count" (1-7, raw count) and
# "Risk Score" (= 100 / count) — they're a deterministic, perfectly-collinear
# pair, not two independent variables, so exactly one gets modeled and the
# other is dropped from the features automatically (see DV_RELATED_COLUMNS)
# rather than left in as a predictor, which would leak the answer.
#
# Defaulting to the raw count: "Risk Score" is a nonlinear (1/x) rescaling
# of it that compresses the high-risk end together (100/6=16.7 vs
# 100/7=14.3, barely distinguishable, vs. count's even 6-vs-7 spacing) —
# that compression can change which model looks best and makes RMSE/MAE
# harder to interpret. If you'd rather model "Risk Score" directly (e.g.
# because that's the number that actually gets reported/acted on), just
# switch this one line — everything else below follows automatically.
DV_COLUMN = "Risk Score Count"
DV_RELATED_COLUMNS = ["Risk Score Count", "Risk Score"]

# "regression" for a continuous DV (earnings, debt, a rate, ...), or
# "classification" for a binary DV (0/1 outcome). Everything below switches
# on this one flag — model registry, metrics, and ensembling all follow it.
TASK_TYPE = "regression"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Lasso-based feature selection, run once up front (Step 3.5, below) before
# any of the models in the registries get fit. Set to False to skip it and
# fit every model on the full one-hot feature set instead.
LASSO_FEATURE_SELECTION = True
LASSO_VOTE_THRESHOLD = 0.5   # keep a feature if its coefficient is nonzero in >= this fraction of the M lasso fits


# ────────────────────────────────────────────────────────────────────────
# Model registries — add/remove a model by editing these dicts, nothing
# else in the script needs to change. Both are here regardless of
# TASK_TYPE so switching tasks later is a one-line config change.
# ────────────────────────────────────────────────────────────────────────

REGRESSION_MODELS = {
    "linear_regression": LinearRegression(),
    "lasso": LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10_000),
    "svr_rbf": SVR(kernel="rbf"),
    "random_forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "bagging": BaggingRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_STATE),
}

CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegressionCV(cv=5, max_iter=5_000, random_state=RANDOM_STATE),
    "lasso_logistic": LogisticRegressionCV(
        cv=5, penalty="l1", solver="liblinear", max_iter=5_000, random_state=RANDOM_STATE
    ),
    "svc_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "bagging": BaggingClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=300, random_state=RANDOM_STATE),
}


# ────────────────────────────────────────────────────────────────────────
# Step 0 — diagnostics: catch bad UNITID columns/files before pandas does,
# with an error that actually says what's wrong
# ────────────────────────────────────────────────────────────────────────

def diagnose_completed_files(completed_dir: Path, n_datasets: int) -> None:
    """Pre-flight check, run before anything else touches these files.
    pandas' own error when index_col="UNITID" can't find that column
    (ValueError: Index UNITID invalid) doesn't say what the header actually
    contained, so this re-reads just the header first and raises a message
    that does — catches a stray BOM, trailing whitespace, or case mismatch
    (e.g. 'UNITID ', '\\ufeffUNITID', 'unitid') immediately, rather than
    several steps later inside a stack trace from inside read_csv.

    Also checks for duplicate UNITID values and for the M files disagreeing
    on which institutions they contain — both would otherwise corrupt the
    shared train/test split silently rather than raising anything: the rest
    of this script assumes one row per UNITID and the same set of UNITIDs
    in every one of the M datasets (see make_shared_split's docstring), so
    either problem needs to be caught here, before it quietly produces
    wrong results downstream instead of an error.
    """
    unitid_sets = []
    for i in range(n_datasets):
        path = completed_dir / f"completed_{i}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — check COMPLETED_DIR and N_DATASETS.")

        header = pd.read_csv(path, nrows=0).columns.tolist()
        if "UNITID" not in header:
            raise ValueError(
                f"{path.name}: no column named exactly 'UNITID' found.\n"
                f"Actual columns: {header!r}\n"
                f"Check for a stray BOM, trailing/leading whitespace, or case "
                f"mismatch in the header (e.g. 'UNITID ', 'unitid', '\\ufeffUNITID')."
            )

        unitid = pd.read_csv(path, usecols=["UNITID"])["UNITID"]
        dupes = unitid[unitid.duplicated(keep=False)]
        if len(dupes):
            example = dupes.unique()[:10].tolist()
            raise ValueError(
                f"{path.name}: {len(dupes)} rows share "
                f"{len(dupes.unique())} duplicated UNITID values (e.g. {example}). "
                f"A duplicated index breaks the shared train/test split (make_shared_split) "
                f"and the feature alignment (align_feature_columns) silently rather than "
                f"raising — dedupe or investigate before proceeding."
            )
        unitid_sets.append(set(unitid))

    reference = unitid_sets[0]
    for i, s in enumerate(unitid_sets[1:], start=1):
        if s != reference:
            only_in_0 = reference - s
            only_in_i = s - reference
            raise ValueError(
                f"completed_0.csv and completed_{i}.csv don't contain the same set of "
                f"institutions (ensembling across the M fits requires matched rows). "
                f"{len(only_in_0)} UNITIDs only in file 0, {len(only_in_i)} only in file {i} "
                f"(e.g. {list(only_in_0 or only_in_i)[:10]})."
            )


# ────────────────────────────────────────────────────────────────────────
# Step 1 — load the M completed datasets, drop institutions missing the DV
# ────────────────────────────────────────────────────────────────────────

def load_completed_datasets(completed_dir: Path, n_datasets: int) -> list[pd.DataFrame]:
    """UNITID doesn't have to be the first column — index_col="UNITID" finds
    it by name, not position — but it does need to be read back as the
    index so the same institution lines up across all M datasets."""
    return [
        pd.read_csv(completed_dir / f"completed_{i}.csv", index_col="UNITID")
        for i in range(n_datasets)
    ]


def drop_missing_dv(datasets: list[pd.DataFrame], dv_column: str) -> list[pd.DataFrame]:
    """The DV is a single non-imputed copy riding along in every one of the
    M datasets, so it's identical row-for-row across all of them — any
    institution missing it is missing it everywhere. Dropping those rows
    (there's nothing to train or score against for them) using one shared
    mask keeps the same institutions in every dataset; if that drops a lot
    of rows, that's worth knowing before modeling, so it's reported rather
    than done silently.
    """
    missing_mask = datasets[0][dv_column].isna()
    n_missing = int(missing_mask.sum())
    if n_missing:
        print(f"Dropped {n_missing} of {len(missing_mask)} institutions with no {dv_column!r} value.")
    keep_index = datasets[0].index[~missing_mask]
    return [df.loc[keep_index] for df in datasets]


# ────────────────────────────────────────────────────────────────────────
# Step 2 — one train/test split, reused across all M datasets
# ────────────────────────────────────────────────────────────────────────

def make_shared_split(index: pd.Index, test_size: float, random_state: int) -> tuple[pd.Index, pd.Index]:
    """The same institutions must be in train vs. test across every one of
    the M datasets — otherwise "average the M predictions for this test row"
    doesn't mean anything, because the M models wouldn't have been tested
    on matched rows. Splitting the shared UNITID index once, up front, and
    reusing it for every dataset is what guarantees that."""
    return train_test_split(index, test_size=test_size, random_state=random_state)


# ────────────────────────────────────────────────────────────────────────
# Step 3 — features: one-hot encode what MICE left as categorical, align
# the resulting columns across all M datasets
# ────────────────────────────────────────────────────────────────────────

def to_model_matrix(
    df: pd.DataFrame, dv_column: str, dv_related_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """mice_pipeline.py deliberately left ordinal columns as plain numeric
    and nominal/binary columns as pandas 'category' dtype (so MICE itself
    treated them correctly — see that script's docstrings). Traditional
    sklearn estimators need everything numeric, so nominal/binary columns
    get one-hot encoded here, at the modeling stage — with drop_first=True,
    for the exact collinearity reason documented throughout mice_pipeline.py
    and README_MICE.md: a full dummy set for an already-binary or nominal
    column is redundant and destabilizes linear/regularized models the same
    way it would have destabilized MICE's internal regressions.

    All of dv_related_columns (not just the chosen dv_column) are dropped
    from the features — "Risk Score"/"Risk Score Count" are a deterministic
    function of each other, so whichever one isn't the DV would otherwise
    leak straight into the model as a near-perfect predictor.
    """
    y = df[dv_column]
    X = pd.get_dummies(df.drop(columns=dv_related_columns), drop_first=True)
    return X, y


def align_feature_columns(
    train_frames: list[pd.DataFrame], test_frames: list[pd.DataFrame]
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """One-hot encoding is done per-dataset, so if a rare category happens
    to not appear in one imputation's realized values (or only appears in
    that imputation's test split), its dummy column could be missing there.
    Reindexing every train/test frame to the union of columns across all M
    (filling absent ones with 0) guarantees every model sees an identical
    feature space regardless of which imputation it was fit on — required
    for the M predictions to be directly comparable/averageable."""
    all_cols = sorted(set().union(*(X.columns for X in train_frames)))
    aligned_train = [X.reindex(columns=all_cols, fill_value=0) for X in train_frames]
    aligned_test = [X.reindex(columns=all_cols, fill_value=0) for X in test_frames]
    return aligned_train, aligned_test


def build_train_test_matrices(
    datasets: list[pd.DataFrame],
    dv_column: str,
    dv_related_columns: list[str],
    train_idx: pd.Index,
    test_idx: pd.Index,
) -> tuple[list[pd.DataFrame], list[pd.Series], list[pd.DataFrame], list[pd.Series]]:
    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
    for df in datasets:
        X, y = to_model_matrix(df, dv_column, dv_related_columns)
        X_train_list.append(X.loc[train_idx])
        y_train_list.append(y.loc[train_idx])
        X_test_list.append(X.loc[test_idx])
        y_test_list.append(y.loc[test_idx])
    X_train_list, X_test_list = align_feature_columns(X_train_list, X_test_list)
    return X_train_list, y_train_list, X_test_list, y_test_list


# ────────────────────────────────────────────────────────────────────────
# Step 3.5 — optional Lasso feature-selection pass: shrink the one-hot
# feature set down before SVM/SVR/random forest/bagging/boosting ever see
# it. This is separate from the "lasso" entry in the model registries above
# — that one reports Lasso's own predictive score; this one just uses
# Lasso's coefficient shrinkage to cut the feature count.
# ────────────────────────────────────────────────────────────────────────

def select_features_with_lasso(
    X_train_list: list[pd.DataFrame],
    y_train_list: list[pd.Series],
    task_type: str,
    vote_threshold: float,
) -> list[str]:
    """Fits an L1-penalized model (LassoCV for regression, L1 LogisticRegressionCV
    for classification) separately on each of the M training sets — same
    one-model-per-imputation pattern used everywhere else in this script —
    and keeps a feature only if it got a nonzero coefficient in at least
    `vote_threshold` of the M fits. This stability-selection-style vote is
    safer than running Lasso on a single imputation (which could keep/drop
    a feature depending on which imputation happened to be used) and smaller
    than taking the union of nonzero features across all M fits.
    """
    n_datasets = len(X_train_list)
    all_columns = X_train_list[0].columns
    selected_counts = pd.Series(0, index=all_columns)

    for X_tr, y_tr in zip(X_train_list, y_train_list):
        X_scaled = StandardScaler().fit_transform(X_tr)
        if task_type == "classification":
            selector = LogisticRegressionCV(
                cv=5, penalty="l1", solver="liblinear", max_iter=5_000, random_state=RANDOM_STATE
            )
        else:
            selector = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10_000)
        selector.fit(X_scaled, y_tr)
        nonzero = all_columns[np.abs(np.ravel(selector.coef_)) > 1e-10]
        selected_counts.loc[nonzero] += 1

    keep = selected_counts[selected_counts >= vote_threshold * n_datasets].index.tolist()
    if not keep:
        print("Lasso feature selection kept 0 features (regularization zeroed out everything) "
              "— falling back to the full feature set.")
        return list(all_columns)

    print(f"Lasso feature selection: kept {len(keep)} of {len(all_columns)} features "
          f"(nonzero in >= {vote_threshold:.0%} of {n_datasets} imputations).")
    return keep


def reduce_to_selected_features(
    X_train_list: list[pd.DataFrame], X_test_list: list[pd.DataFrame], selected_features: list[str]
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    return (
        [X.loc[:, selected_features] for X in X_train_list],
        [X.loc[:, selected_features] for X in X_test_list],
    )


# ────────────────────────────────────────────────────────────────────────
# Step 4 — fit each model on each of the M datasets, ensemble predictions
# ────────────────────────────────────────────────────────────────────────

def score_regression(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
    }


def score_classification(y_true, y_proba) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
    }


def fit_predict_one(estimator, X_train, y_train, X_test, task_type: str) -> np.ndarray:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])
    pipe.fit(X_train, y_train)
    if task_type == "classification":
        return pipe.predict_proba(X_test)[:, 1]
    return pipe.predict(X_test)


def run_models_across_imputations(
    models: dict,
    X_train_list: list[pd.DataFrame],
    y_train_list: list[pd.Series],
    X_test_list: list[pd.DataFrame],
    y_test_list: list[pd.Series],
    task_type: str,
) -> pd.DataFrame:
    """For each model: fit on each of the M (X_train, y_train) pairs,
    predict on the matching X_test, then (a) average the M predictions and
    score that ensembled prediction against the shared y_test, and
    (b) score each of the M individual fits separately to report how much
    the metric moves across imputations — see the module docstring for why
    both numbers are worth having."""
    score_fn = score_classification if task_type == "classification" else score_regression
    y_test_reference = y_test_list[0]  # identical across all M by construction (same DV, same split)

    rows = []
    for name, estimator in models.items():
        per_imputation_preds = []
        per_imputation_scores = []
        for X_tr, y_tr, X_te, y_te in zip(X_train_list, y_train_list, X_test_list, y_test_list):
            pred = fit_predict_one(estimator, X_tr, y_tr, X_te, task_type)
            per_imputation_preds.append(pred)
            per_imputation_scores.append(score_fn(y_te, pred))

        ensembled_pred = np.mean(per_imputation_preds, axis=0)
        ensembled_scores = score_fn(y_test_reference, ensembled_pred)

        per_imputation_df = pd.DataFrame(per_imputation_scores)
        row = {"model": name}
        for metric in ensembled_scores:
            row[f"ensembled_{metric}"] = ensembled_scores[metric]
            row[f"per_imputation_{metric}_mean"] = per_imputation_df[metric].mean()
            row[f"per_imputation_{metric}_std"] = per_imputation_df[metric].std()
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


# ────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────

def main():
    diagnose_completed_files(COMPLETED_DIR, N_DATASETS)
    completed = load_completed_datasets(COMPLETED_DIR, N_DATASETS)
    completed = drop_missing_dv(completed, DV_COLUMN)

    train_idx, test_idx = make_shared_split(completed[0].index, TEST_SIZE, RANDOM_STATE)
    X_train_list, y_train_list, X_test_list, y_test_list = build_train_test_matrices(
        completed, DV_COLUMN, DV_RELATED_COLUMNS, train_idx, test_idx
    )
    print(f"Train: {len(train_idx)} institutions  |  Test: {len(test_idx)}  |  "
          f"Features after one-hot + alignment: {X_train_list[0].shape[1]}")

    if LASSO_FEATURE_SELECTION:
        selected_features = select_features_with_lasso(
            X_train_list, y_train_list, TASK_TYPE, LASSO_VOTE_THRESHOLD
        )
        X_train_list, X_test_list = reduce_to_selected_features(X_train_list, X_test_list, selected_features)

    models = CLASSIFICATION_MODELS if TASK_TYPE == "classification" else REGRESSION_MODELS
    results = run_models_across_imputations(
        models, X_train_list, y_train_list, X_test_list, y_test_list, TASK_TYPE
    )

    pd.set_option("display.width", 120)
    print(results.round(4))
    results.to_csv(COMPLETED_DIR / "model_comparison.csv")


if __name__ == "__main__":
    main()
