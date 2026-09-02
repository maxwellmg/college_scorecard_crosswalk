"""
College Scorecard — post-MICE predictive modeling
====================================================
Companion to mice_pipeline.py. Loads the M completed datasets that script
produced, attaches a fully-observed dependent variable (one copy — not
imputed, so it's identical across all M datasets by construction), and fits
a handful of different model types to compare predictive accuracy.

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
run against your actual completed_*.csv files or real DV, since neither
exists yet in this environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

COMPLETED_DIR = Path("mice_output")   # matches mice_pipeline.py's OUTPUT_DIR
N_DATASETS = 5                        # must match mice_pipeline.py's N_DATASETS

DV_CSV = "dependent_variable.csv"     # your file: one row per UNITID, one DV column
DV_COLUMN = "YOUR_DV_COLUMN_NAME"     # <-- placeholder, edit this

# "regression" for a continuous DV (earnings, debt, a rate, ...), or
# "classification" for a binary DV (0/1 outcome). Everything below switches
# on this one flag — model registry, metrics, and ensembling all follow it.
TASK_TYPE = "regression"

TEST_SIZE = 0.2
RANDOM_STATE = 42


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
}

CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegressionCV(cv=5, max_iter=5_000, random_state=RANDOM_STATE),
    "lasso_logistic": LogisticRegressionCV(
        cv=5, penalty="l1", solver="liblinear", max_iter=5_000, random_state=RANDOM_STATE
    ),
    "svc_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
}


# ────────────────────────────────────────────────────────────────────────
# Step 1 — load the M completed datasets + attach the (single-copy) DV
# ────────────────────────────────────────────────────────────────────────

def load_completed_datasets(completed_dir: Path, n_datasets: int) -> list[pd.DataFrame]:
    """Each completed_{i}.csv has UNITID as its first column (mice_pipeline.py
    writes it with index=True) — read it back as the index so it lines up
    with the DV file on the join in the next step."""
    return [
        pd.read_csv(completed_dir / f"completed_{i}.csv", index_col="UNITID")
        for i in range(n_datasets)
    ]


def attach_dependent_variable(
    datasets: list[pd.DataFrame], dv_csv: str | Path, dv_column: str
) -> list[pd.DataFrame]:
    """Joins the same DV column onto every one of the M datasets — it's a
    single, non-imputed copy, so this is an ordinary join repeated M times,
    not anything MICE-related. Institutions missing the DV are dropped
    (inner join) since there's nothing to train or score against for them;
    if that drops a lot of rows, that's worth knowing before modeling, so
    it's reported rather than done silently.
    """
    dv = pd.read_csv(dv_csv, index_col="UNITID")[[dv_column]].dropna()
    merged = []
    for df in datasets:
        m = df.join(dv, how="inner")
        merged.append(m)
    n_before, n_after = len(datasets[0]), len(merged[0])
    if n_after < n_before:
        print(f"Dropped {n_before - n_after} of {n_before} institutions with no {dv_column!r} value.")
    return merged


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

def to_model_matrix(df: pd.DataFrame, dv_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """mice_pipeline.py deliberately left ordinal columns as plain numeric
    and nominal/binary columns as pandas 'category' dtype (so MICE itself
    treated them correctly — see that script's docstrings). Traditional
    sklearn estimators need everything numeric, so nominal/binary columns
    get one-hot encoded here, at the modeling stage — with drop_first=True,
    for the exact collinearity reason documented throughout mice_pipeline.py
    and README_MICE.md: a full dummy set for an already-binary or nominal
    column is redundant and destabilizes linear/regularized models the same
    way it would have destabilized MICE's internal regressions.
    """
    y = df[dv_column]
    X = pd.get_dummies(df.drop(columns=[dv_column]), drop_first=True)
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
    merged: list[pd.DataFrame], dv_column: str, train_idx: pd.Index, test_idx: pd.Index
) -> tuple[list[pd.DataFrame], list[pd.Series], list[pd.DataFrame], list[pd.Series]]:
    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
    for df in merged:
        X, y = to_model_matrix(df, dv_column)
        X_train_list.append(X.loc[train_idx])
        y_train_list.append(y.loc[train_idx])
        X_test_list.append(X.loc[test_idx])
        y_test_list.append(y.loc[test_idx])
    X_train_list, X_test_list = align_feature_columns(X_train_list, X_test_list)
    return X_train_list, y_train_list, X_test_list, y_test_list


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
    completed = load_completed_datasets(COMPLETED_DIR, N_DATASETS)
    merged = attach_dependent_variable(completed, DV_CSV, DV_COLUMN)

    train_idx, test_idx = make_shared_split(merged[0].index, TEST_SIZE, RANDOM_STATE)
    X_train_list, y_train_list, X_test_list, y_test_list = build_train_test_matrices(
        merged, DV_COLUMN, train_idx, test_idx
    )
    print(f"Train: {len(train_idx)} institutions  |  Test: {len(test_idx)}  |  "
          f"Features after one-hot + alignment: {X_train_list[0].shape[1]}")

    models = CLASSIFICATION_MODELS if TASK_TYPE == "classification" else REGRESSION_MODELS
    results = run_models_across_imputations(
        models, X_train_list, y_train_list, X_test_list, y_test_list, TASK_TYPE
    )

    pd.set_option("display.width", 120)
    print(results.round(4))
    results.to_csv(COMPLETED_DIR / "model_comparison.csv")


if __name__ == "__main__":
    main()
