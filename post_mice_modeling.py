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

Dependencies: pandas, numpy, scikit-learn, matplotlib (`pip install scikit-learn matplotlib`).
Validated end-to-end against a synthetic stand-in for mice_pipeline.py's
output (see the bottom of this file's test run in conversation) — not yet
run against your actual completed_*.csv files, since this environment
doesn't have them.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
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
    roc_curve,
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

# Some feature columns still have leftover NaNs (outside MICE's imputation
# set) rather than every column being fully imputed. True drops any
# institution with a missing value in any feature column — the UNION of
# such institutions across all M completed_#.csv files, dropped from every
# one of them, so all M stay aligned on the same institutions (see
# drop_rows_with_missing_features). False leaves diagnose_completed_files'
# hard failure in place instead, so a NaN can't reach a model silently.
DROP_ROWS_WITH_MISSING_FEATURES = True

# Column names to drop entirely (not just the rows with a missing value in
# them) before any missing-value checks or modeling — for a column that's
# bad outright rather than just missing a few values, this keeps far more
# institutions in play than DROP_ROWS_WITH_MISSING_FEATURES would for the
# same column. A name not present in a given completed_#.csv is skipped
# there without error.
COLUMNS_TO_DROP: list[str] = []

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

# Saves a model-comparison chart + a diagnostic plot for the best model to
# COMPLETED_DIR as PNGs once every model has finished (Step 5, below).
MAKE_PLOTS = True

# A few roles from a validated, colorblind-safe default palette — just the
# ones these plots need (one accent color; everything else is neutral ink/
# gridlines), so there's no cycling through an arbitrary color list.
PLOT_COLORS = {
    "accent": "#2a78d6",
    "ink": "#0b0b0b",
    "muted": "#898781",
    "grid": "#e1e0d9",
}

# At small N (e.g. ~78 institutions), a single 80/20 holdout test set is a
# noisy, one-shot estimate — these two add cross-validated estimates that
# use far more of the data per fit. Both reuse whichever feature set the
# main holdout run above ended up with (post-Lasso-reduction, if enabled)
# rather than re-running feature selection inside every fold — see
# run_loocv_across_imputations' docstring for why that's a documented
# simplification rather than a fully nested CV. Both can be slow at a lot
# of models/imputations — turn either off if the runtime's a problem.
RUN_LOOCV = True
RUN_BOOTSTRAP = True
N_BOOTSTRAP_ITERS = 200   # out-of-bag bootstrap replicates; runtime scales linearly with this


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

def diagnose_completed_files(
    completed_dir: Path,
    n_datasets: int,
    dv_related_columns: list[str],
    drop_rows_with_missing_features: bool,
    columns_to_drop: list[str],
) -> None:
    """Pre-flight check, run before anything else touches these files.
    pandas' own error when index_col="UNITID" can't find that column
    (ValueError: Index UNITID invalid) doesn't say what the header actually
    contained, so this re-reads just the header first and raises a message
    that does — catches a stray BOM, trailing whitespace, or case mismatch
    (e.g. 'UNITID ', '\\ufeffUNITID', 'unitid') immediately, rather than
    several steps later inside a stack trace from inside read_csv. The same
    check runs for dv_related_columns (DV_COLUMN and whatever else it's
    paired with, e.g. 'Risk Score'/'Risk Score Count') — those get looked
    up with plain `df[col]` later (in drop_missing_dv / to_model_matrix),
    which raises a bare `KeyError: 'Risk Score Count'` with no indication
    of what the header actually contained, same root cause as the UNITID
    case and just as easy to get from a stray typo/space/case mismatch.

    Also checks every feature column (everything except UNITID,
    dv_related_columns — a missing DV is handled separately and on purpose,
    by drop_missing_dv dropping that row — and columns_to_drop, which get
    dropped outright before modeling regardless of what's in them) for
    leftover NaNs. A completed_#.csv
    implies MICE filled in every column, so a NaN surviving here almost
    always means a column sat outside mice_pipeline.py's imputation set
    (e.g. not listed in its MODEL_VARS, or passively derived from a column
    that was) — it'll otherwise reach a model as a bare, column-agnostic
    "Input X contains NaN" from inside StandardScaler/LassoCV. With
    drop_rows_with_missing_features=True (DROP_ROWS_WITH_MISSING_FEATURES in
    config), this is reported as a note — load_completed_datasets +
    drop_rows_with_missing_features will drop the affected institutions
    automatically — rather than raised as fatal.

    Also flags duplicate UNITID values (load_completed_datasets dedupes
    these automatically — see dedupe_by_unitid — so this is reported as a
    note, not an error) and checks for the M files disagreeing on which
    institutions they contain — that one IS still fatal, since ensembling
    across the M fits requires the same institutions, in the same positions,
    in every one of the M datasets (see make_shared_split's docstring), and
    there's no sensible automatic fix for a genuine mismatch in membership.
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

        missing_dv_cols = [col for col in dv_related_columns if col not in header]
        if missing_dv_cols:
            raise ValueError(
                f"{path.name}: column(s) {missing_dv_cols!r} from DV_COLUMN/DV_RELATED_COLUMNS "
                f"not found.\nActual columns: {header!r}\n"
                f"Check for a typo, trailing/leading whitespace, or case mismatch against "
                f"DV_RELATED_COLUMNS in the config section."
            )

        df = pd.read_csv(path)

        feature_cols = [
            c for c in df.columns if c != "UNITID" and c not in dv_related_columns and c not in columns_to_drop
        ]
        na_counts = df[feature_cols].isna().sum()
        na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
        if len(na_counts):
            listing = "\n".join(
                f"  {col}: {n} missing ({n / len(df):.1%} of {len(df)} rows)"
                for col, n in na_counts.items()
            )
            if drop_rows_with_missing_features:
                print(
                    f"NOTE: {path.name}: {len(na_counts)} feature column(s) still have missing "
                    f"values — affected institutions will be dropped automatically "
                    f"(DROP_ROWS_WITH_MISSING_FEATURES=True):\n{listing}"
                )
            else:
                raise ValueError(
                    f"{path.name}: {len(na_counts)} feature column(s) still have missing values:\n"
                    f"{listing}\n"
                    f"Every column in a completed_#.csv should be fully imputed — check whether "
                    f"these were included in mice_pipeline.py's imputation set, or set "
                    f"DROP_ROWS_WITH_MISSING_FEATURES = True to drop these institutions instead."
                )

        unitid = df["UNITID"]
        dupes = unitid[unitid.duplicated(keep=False)]
        if len(dupes):
            print(
                f"NOTE: {path.name} has {len(dupes)} rows across {len(dupes.unique())} "
                f"duplicated UNITID values — the first occurrence of each will be kept "
                f"and the rest dropped automatically when this file is loaded."
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
# Step 1 — load the M completed datasets (deduping by UNITID), drop
# institutions missing the DV or (optionally) any feature value
# ────────────────────────────────────────────────────────────────────────

def dedupe_by_unitid(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with a duplicate UNITID index, keeping the first
    occurrence. Left alone, a duplicated index label blows up row counts
    wherever this script does `.loc[some_index]` (every matching row comes
    back for each occurrence of a repeated label, in both the lookup and
    the frame) — make_shared_split and align_feature_columns both rely on
    one row per UNITID, so this collapses to that before either runs.
    """
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_after = len(df)
    if n_after < n_before:
        print(f"Deduped {n_before - n_after} duplicate-UNITID rows (kept first occurrence).")
    return df


def load_completed_datasets(completed_dir: Path, n_datasets: int) -> list[pd.DataFrame]:
    """UNITID doesn't have to be the first column — index_col="UNITID" finds
    it by name, not position — but it does need to be read back as the
    index so the same institution lines up across all M datasets."""
    datasets = [
        pd.read_csv(completed_dir / f"completed_{i}.csv", index_col="UNITID")
        for i in range(n_datasets)
    ]
    return [dedupe_by_unitid(df) for df in datasets]


def drop_configured_columns(datasets: list[pd.DataFrame], columns_to_drop: list[str]) -> list[pd.DataFrame]:
    """Drops COLUMNS_TO_DROP entirely from every one of the M datasets. For
    a column that's bad outright (not just missing a handful of values),
    dropping the column keeps far more institutions in play than
    drop_rows_with_missing_features would for that same column — this runs
    first, before any missing-value handling, so a configured-bad column
    never costs a row elsewhere in the pipeline.
    """
    present = [c for c in columns_to_drop if c in datasets[0].columns]
    if present:
        print(f"Dropping {len(present)} configured column(s) from every dataset: {present}")
    return [df.drop(columns=present, errors="ignore") for df in datasets]


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


def drop_rows_with_missing_features(
    datasets: list[pd.DataFrame], dv_related_columns: list[str]
) -> list[pd.DataFrame]:
    """Drops any institution with a missing value in any feature column
    (everything except dv_related_columns — a missing DV is drop_missing_dv's
    job, not this one). An institution counts as affected if it's missing a
    feature in ANY of the M datasets, even if the other M-1 are fine for it
    — the union, not the intersection, is dropped from every one of the M
    datasets, since make_shared_split and the per-fold loops elsewhere in
    this script all require the same institutions to be present in every
    one of the M datasets.
    """
    bad_ids = set()
    for df in datasets:
        feature_cols = [c for c in df.columns if c not in dv_related_columns]
        bad_ids.update(df.index[df[feature_cols].isna().any(axis=1)])

    if bad_ids:
        print(f"Dropped {len(bad_ids)} institutions with a missing feature value in at least "
              f"one imputation (DROP_ROWS_WITH_MISSING_FEATURES=True): {sorted(bad_ids)[:10]}"
              f"{', ...' if len(bad_ids) > 10 else ''}")
    keep_index = datasets[0].index[~datasets[0].index.isin(bad_ids)]
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


def build_full_matrices(
    datasets: list[pd.DataFrame],
    dv_column: str,
    dv_related_columns: list[str],
    feature_columns: list[str] | None = None,
) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    """Same one-hot-encode-then-align process as build_train_test_matrices,
    but over the full index rather than a train/test split — LOOCV and the
    bootstrap (Step 5.5, below) each carve up these full matrices into many
    different train/test partitions themselves, so they need the complete
    feature matrix once rather than one fixed split of it.

    feature_columns, when given, reindexes down to exactly that column set
    (e.g. whatever select_features_with_lasso already picked for the main
    holdout run) so LOOCV/bootstrap score the same feature space as the
    holdout comparison rather than re-deriving their own.
    """
    X_list, y_list = [], []
    for df in datasets:
        X, y = to_model_matrix(df, dv_column, dv_related_columns)
        X_list.append(X)
        y_list.append(y)
    all_cols = sorted(set().union(*(X.columns for X in X_list)))
    X_list = [X.reindex(columns=all_cols, fill_value=0) for X in X_list]
    if feature_columns is not None:
        X_list = [X.reindex(columns=feature_columns, fill_value=0) for X in X_list]
    return X_list, y_list


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
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """For each model: fit on each of the M (X_train, y_train) pairs,
    predict on the matching X_test, then (a) average the M predictions and
    score that ensembled prediction against the shared y_test, and
    (b) score each of the M individual fits separately to report how much
    the metric moves across imputations — see the module docstring for why
    both numbers are worth having. Also returns each model's ensembled
    prediction (not just its scores) — Step 5's diagnostic plot needs the
    actual predicted values, not only the summary metrics in the table."""
    score_fn = score_classification if task_type == "classification" else score_regression
    y_test_reference = y_test_list[0]  # identical across all M by construction (same DV, same split)

    rows = []
    ensembled_predictions = {}
    for name, estimator in models.items():
        per_imputation_preds = []
        per_imputation_scores = []
        for X_tr, y_tr, X_te, y_te in zip(X_train_list, y_train_list, X_test_list, y_test_list):
            pred = fit_predict_one(estimator, X_tr, y_tr, X_te, task_type)
            per_imputation_preds.append(pred)
            per_imputation_scores.append(score_fn(y_te, pred))

        ensembled_pred = np.mean(per_imputation_preds, axis=0)
        ensembled_predictions[name] = ensembled_pred
        ensembled_scores = score_fn(y_test_reference, ensembled_pred)

        per_imputation_df = pd.DataFrame(per_imputation_scores)
        row = {"model": name}
        for metric in ensembled_scores:
            row[f"ensembled_{metric}"] = ensembled_scores[metric]
            row[f"per_imputation_{metric}_mean"] = per_imputation_df[metric].mean()
            row[f"per_imputation_{metric}_std"] = per_imputation_df[metric].std()
        rows.append(row)

    return pd.DataFrame(rows).set_index("model"), ensembled_predictions


# ────────────────────────────────────────────────────────────────────────
# Step 5 — plots: a model-comparison chart across every metric, plus a
# diagnostic plot for whichever model scored best
# ────────────────────────────────────────────────────────────────────────

def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PLOT_COLORS["muted"])
    ax.spines["bottom"].set_color(PLOT_COLORS["muted"])
    ax.tick_params(colors=PLOT_COLORS["muted"])
    ax.yaxis.grid(True, color=PLOT_COLORS["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_model_comparison(results: pd.DataFrame, task_type: str, out_path: Path) -> None:
    """One panel per metric, bars ranked by the ensembled score, with an
    error bar showing the spread of the M per-imputation scores — the
    imputation-sensitivity read described in the module docstring. Each
    panel is a single series (one model's score per bar), so it gets one
    accent color throughout rather than a categorical palette — there's no
    second series here for color to distinguish.
    """
    metrics = ["r2", "rmse", "mae"] if task_type == "regression" else ["accuracy", "roc_auc", "f1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    for ax, metric in zip(axes, metrics):
        ranked = results.sort_values(f"ensembled_{metric}", ascending=False)
        ax.bar(
            ranked.index,
            ranked[f"ensembled_{metric}"],
            yerr=ranked[f"per_imputation_{metric}_std"],
            capsize=4,
            color=PLOT_COLORS["accent"],
            ecolor=PLOT_COLORS["muted"],
        )
        ax.set_title(metric.upper(), color=PLOT_COLORS["ink"])
        ax.tick_params(axis="x", rotation=40)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        _style_axis(ax)
    fig.suptitle("Model comparison — ensembled score ± per-imputation spread", color=PLOT_COLORS["ink"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved model comparison plot to {out_path}")


def plot_best_model_diagnostic(
    best_model_name: str, y_true, y_pred: np.ndarray, task_type: str, out_path: Path
) -> None:
    """Predicted-vs-actual (regression) or an ROC curve (classification)
    for whichever model came out on top by the primary metric — the
    comparison chart above says how much better; this shows what its
    predictions actually look like against the truth."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if task_type == "classification":
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ax.plot(fpr, tpr, color=PLOT_COLORS["accent"], linewidth=2)
        ax.plot([0, 1], [0, 1], color=PLOT_COLORS["muted"], linewidth=1, linestyle="--")
        ax.set_xlabel("False positive rate", color=PLOT_COLORS["ink"])
        ax.set_ylabel("True positive rate", color=PLOT_COLORS["ink"])
        ax.set_title(f"ROC curve — {best_model_name}", color=PLOT_COLORS["ink"])
    else:
        ax.scatter(y_true, y_pred, color=PLOT_COLORS["accent"], alpha=0.6, s=28, edgecolor="none")
        lo, hi = min(np.min(y_true), y_pred.min()), max(np.max(y_true), y_pred.max())
        ax.plot([lo, hi], [lo, hi], color=PLOT_COLORS["muted"], linewidth=1, linestyle="--")
        ax.set_xlabel("Actual", color=PLOT_COLORS["ink"])
        ax.set_ylabel("Predicted", color=PLOT_COLORS["ink"])
        ax.set_title(f"Predicted vs. actual — {best_model_name}", color=PLOT_COLORS["ink"])
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved best-model diagnostic plot to {out_path}")


# ────────────────────────────────────────────────────────────────────────
# Step 5.5 — LOOCV and out-of-bag bootstrap: more data-efficient evaluation
# for a small N (e.g. ~78 institutions), where the single 80/20 holdout
# above is a noisy, one-shot estimate. Both work over the full feature
# matrices from build_full_matrices and ensemble across the M imputations
# the same way as everywhere else in this script.
# ────────────────────────────────────────────────────────────────────────

def run_loocv_across_imputations(
    models: dict,
    X_list: list[pd.DataFrame],
    y_list: list[pd.Series],
    task_type: str,
) -> pd.DataFrame:
    """Leave-one-out CV: for each institution, fit on the other N-1 (once
    per imputation), average the M predictions for that one held-out
    institution, then move on. At N~78 this trains on ~98.7% of the data
    per fit, vs. 80% for the holdout split — the tradeoff is one prediction
    per fold rather than many, so there's no per-fold score to report (a
    single point has no R²/accuracy); only the pooled score across all N
    held-out predictions is reported here.

    NOTE: this reuses whichever feature columns the caller passes in (the
    main holdout run's post-Lasso-reduction feature set, by convention) for
    every fold rather than re-running select_features_with_lasso inside
    each one — a fully nested CV would redo feature selection per fold.
    Skipping that here is a deliberate simplification for a first pass at
    N=78 (78 refits of LassoCV per imputation, on top of everything else,
    for a marginal rigor gain); it gives LOOCV's accuracy numbers a small
    optimistic bias worth keeping in mind if they end up driving a decision.
    """
    index = X_list[0].index
    score_fn = score_classification if task_type == "classification" else score_regression

    rows = []
    for name, estimator in models.items():
        loo_preds = pd.Series(index=index, dtype=float)
        for held_out in index:
            train_idx = index.drop(held_out)
            test_idx = [held_out]
            per_imputation_preds = [
                fit_predict_one(estimator, X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], task_type)[0]
                for X, y in zip(X_list, y_list)
            ]
            loo_preds.loc[held_out] = np.mean(per_imputation_preds)

        y_true = y_list[0]  # identical across all M by construction
        scores = score_fn(y_true, loo_preds)
        rows.append({"model": name, **{f"loocv_{metric}": value for metric, value in scores.items()}})

    return pd.DataFrame(rows).set_index("model")


def run_bootstrap_across_imputations(
    models: dict,
    X_list: list[pd.DataFrame],
    y_list: list[pd.Series],
    task_type: str,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    """Out-of-bag bootstrap: each iteration draws N institutions with
    replacement (same draw shared across all M imputations, same pattern as
    make_shared_split) as the training set — whichever institutions weren't
    drawn at all (~37% of N on average) are that iteration's out-of-bag test
    set. A model is fit on the draw once per imputation, the M predictions
    for that iteration's OOB institutions are averaged, and the (true,
    predicted) pairs are pooled across all n_bootstrap iterations before
    scoring once at the end, rather than averaged per institution first.

    This is the plain OOB bootstrap estimate, not the bias-corrected .632+
    estimator (which additionally blends in the apparent/training error
    with fixed weights) — a reasonable first pass, not the most refined
    version of this if these numbers need to be tighter later.
    """
    rng = np.random.default_rng(random_state)
    index = X_list[0].index
    n = len(index)
    score_fn = score_classification if task_type == "classification" else score_regression

    rows = []
    for name, estimator in models.items():
        pooled_true, pooled_pred = [], []
        for _ in range(n_bootstrap):
            boot_idx = index[rng.integers(0, n, size=n)]
            oob_idx = index.difference(boot_idx)
            if len(oob_idx) == 0:
                continue
            per_imputation_preds = [
                fit_predict_one(estimator, X.loc[boot_idx], y.loc[boot_idx], X.loc[oob_idx], task_type)
                for X, y in zip(X_list, y_list)
            ]
            pooled_true.append(y_list[0].loc[oob_idx].to_numpy())
            pooled_pred.append(np.mean(per_imputation_preds, axis=0))

        scores = score_fn(np.concatenate(pooled_true), np.concatenate(pooled_pred))
        rows.append({"model": name, **{f"bootstrap_{metric}": value for metric, value in scores.items()}})

    return pd.DataFrame(rows).set_index("model")


# ────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────

def main():
    diagnose_completed_files(
        COMPLETED_DIR, N_DATASETS, DV_RELATED_COLUMNS, DROP_ROWS_WITH_MISSING_FEATURES, COLUMNS_TO_DROP
    )
    completed = load_completed_datasets(COMPLETED_DIR, N_DATASETS)
    completed = drop_configured_columns(completed, COLUMNS_TO_DROP)
    completed = drop_missing_dv(completed, DV_COLUMN)
    if DROP_ROWS_WITH_MISSING_FEATURES:
        completed = drop_rows_with_missing_features(completed, DV_RELATED_COLUMNS)

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
    results, ensembled_predictions = run_models_across_imputations(
        models, X_train_list, y_train_list, X_test_list, y_test_list, TASK_TYPE
    )

    pd.set_option("display.width", 120)
    print(results.round(4))
    results.to_csv(COMPLETED_DIR / "model_comparison.csv")

    if MAKE_PLOTS:
        plot_model_comparison(results, TASK_TYPE, COMPLETED_DIR / "model_comparison.png")
        primary_metric = "roc_auc" if TASK_TYPE == "classification" else "r2"
        best_model_name = results[f"ensembled_{primary_metric}"].idxmax()
        plot_best_model_diagnostic(
            best_model_name,
            y_test_list[0],
            ensembled_predictions[best_model_name],
            TASK_TYPE,
            COMPLETED_DIR / f"best_model_diagnostic_{best_model_name}.png",
        )

    if RUN_LOOCV or RUN_BOOTSTRAP:
        feature_columns = X_train_list[0].columns.tolist()
        X_full_list, y_full_list = build_full_matrices(completed, DV_COLUMN, DV_RELATED_COLUMNS, feature_columns)
        n_institutions = len(X_full_list[0])

    if RUN_LOOCV:
        print(f"Running LOOCV: {n_institutions} folds x {N_DATASETS} imputations x {len(models)} models...")
        t0 = time.perf_counter()
        loocv_results = run_loocv_across_imputations(models, X_full_list, y_full_list, TASK_TYPE)
        print(f"LOOCV done in {time.perf_counter() - t0:.1f}s")
        print(loocv_results.round(4))
        loocv_results.to_csv(COMPLETED_DIR / "loocv_results.csv")

    if RUN_BOOTSTRAP:
        print(f"Running OOB bootstrap: {N_BOOTSTRAP_ITERS} iterations x {N_DATASETS} imputations x "
              f"{len(models)} models...")
        t0 = time.perf_counter()
        bootstrap_results = run_bootstrap_across_imputations(
            models, X_full_list, y_full_list, TASK_TYPE, N_BOOTSTRAP_ITERS, RANDOM_STATE
        )
        print(f"Bootstrap done in {time.perf_counter() - t0:.1f}s")
        print(bootstrap_results.round(4))
        bootstrap_results.to_csv(COMPLETED_DIR / "bootstrap_results.csv")


if __name__ == "__main__":
    main()
