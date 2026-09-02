# College Scorecard — Post-MICE Modeling Guide

**Purpose:** how to run `post_mice_modeling.py` and how to read what it
prints out. Companion to `README_MICE.md` (the preprocessing stage) and
`mice_pipeline.py` (which produces this script's input).

**Scope:** a first accuracy-and-predictive-power pass across a handful of
model types, using multiple imputation's `M` completed datasets correctly.
It is not a final model, not a causal/inferential analysis, and not the
MCMC/Bayesian treatment you mentioned wanting eventually — that's later,
separate work, deliberately not started here.

---

## 1. Prerequisites

Before running this script, you need:

1. **`mice_pipeline.py` has already run to completion**, producing
   `mice_output/completed_0.csv` … `completed_{M-1}.csv`. Each file has
   `UNITID` as its first column (an index, not a feature) and every other
   surviving column already ordinal-encoded / one-hot-ready per
   `README_MICE.md`.
2. **A dependent-variable CSV** — your own file, one row per institution,
   with a `UNITID` column and whatever outcome column you're predicting
   (e.g. an earnings figure, a completion rate, a binary flag). This value
   is assumed fully observed and identical across all `M` datasets — it
   does **not** go through MICE, per your original scoping of this ("I'm
   adding a dependent variable that does not need to be imputed").
3. `pip install scikit-learn` (pandas/numpy are already required by
   `mice_pipeline.py`).

---

## 2. Config checklist — edit these before running

| Constant | Meaning | Action needed |
|---|---|---|
| `COMPLETED_DIR` | Where the completed datasets live | Must match `mice_pipeline.py`'s `OUTPUT_DIR` |
| `N_DATASETS` | How many completed datasets to load | Must match `mice_pipeline.py`'s `N_DATASETS` (its `M`) |
| `DV_CSV` | Path to your dependent-variable file | **Edit** — placeholder path |
| `DV_COLUMN` | Column name of the outcome in that file | **Edit** — placeholder (`"YOUR_DV_COLUMN_NAME"`) |
| `TASK_TYPE` | `"regression"` or `"classification"` | **Confirm** — picks the model registry, scoring functions, and prediction-ensembling rule (mean vs. probability-mean) |
| `TEST_SIZE` | Held-out share for evaluation | Default `0.2`; raise it if the institution count after joining the DV is small |
| `RANDOM_STATE` | Seed for the split and every model | Default `42`; change only if you want a different split/fit, not to "try for a better score" |

If `COMPLETED_DIR`/`N_DATASETS` don't match what `mice_pipeline.py` actually
produced, the load step in §3 below will raise `FileNotFoundError` on a
missing `completed_i.csv` rather than silently reading a partial `M`.

---

## 3. What the script actually does, step by step

**Step 1 — load + attach the DV** (`load_completed_datasets`,
`attach_dependent_variable`). Reads all `M` completed CSVs with `UNITID` as
the index, then joins your DV column onto each one. This is an ordinary
join done `M` times — nothing MICE-related happens to the DV itself. The
join is `inner`: an institution missing from your DV file drops out of all
`M` datasets, and the script prints how many rows that cost you. If that
number is large relative to your total institution count, that's worth
investigating before trusting the results — it's a silent selection effect
otherwise.

**Step 2 — one shared train/test split** (`make_shared_split`). Computed
**once**, on the post-join institution index, then reused for all `M`
datasets. This is the part that's easy to get wrong: if each of the `M`
datasets got its own random split, the `M` models would be scored on
different held-out institutions, and "average the `M` predictions" would be
averaging predictions for different rows under the same index label —
silently wrong, not just imprecise. One split, reused everywhere, is what
keeps the ensembling in Step 4 meaningful.

**Step 3 — build model matrices** (`to_model_matrix`,
`align_feature_columns`, `build_train_test_matrices`). `mice_pipeline.py`
deliberately left ordinal columns as plain numeric and nominal/binary
columns as pandas `category` dtype — correct for MICE, but traditional
sklearn estimators need everything numeric. This step one-hot encodes the
categorical columns here, at the modeling stage, with `drop_first=True` —
the same collinearity reasoning that's been consistent throughout this
project: a full dummy set for an already-binary or nominal column is
redundant and destabilizes linear/regularized models the same way it would
have destabilized MICE's own internal regressions.

Because one-hot encoding runs separately on each of the `M` datasets, a
rare category could in principle produce a dummy column in one imputation's
data that doesn't appear in another's (or that appears in train but not
test for a given imputation). `align_feature_columns` reindexes every
train/test frame to the union of columns across all `M`, filling absent
ones with 0, so every model — regardless of which imputation it was fit on
— sees an identical feature space. Without this, predictions from different
imputations wouldn't even be guaranteed to correspond to the same
coefficients/features, let alone be averageable.

**Step 4 — fit, ensemble, score** (`fit_predict_one`,
`run_models_across_imputations`). For each model in the active registry:
fit it separately on each of the `M` (X_train, y_train) pairs (each inside
a `Pipeline` with a `StandardScaler`, since SVM and Lasso are scale-
sensitive and scaling a linear/RF model's inputs doesn't hurt it), predict
on the matching `X_test`, then:
- **average the `M` predictions** (for classification, this means
  averaging predicted *probabilities*, not hard labels — probability-
  averaging is the more standard and more information-preserving way to
  ensemble classifiers than majority-voting hard predictions) and score
  that single ensembled prediction against the shared `y_test`, and
- **separately score each of the `M` individual fits**, then report the
  mean and standard deviation of that metric across the `M` fits.

Both numbers get written to every row of the output. See §5 for why you
want both.

---

## 4. Why prediction-averaging instead of Rubin's rules

This directly follows from the "will I be able to use all M iterations"
question earlier in this project. Rubin's rules — the classical way to
combine results across multiply-imputed datasets — pools a model's
*coefficients* and their *standard errors*. That's well-defined for OLS and
logistic regression, where both exist and have known sampling theory. It is
**not** well-defined the same way for Lasso (coefficients are shrunk and
which variables get selected can differ across imputations — there's no
standard "average these" rule) or SVM (no coefficient to pool at all, for
a nonlinear kernel).

Since this script's whole point is comparing several model types side by
side — including Lasso and SVM — it uses one ensembling rule that works
identically for all of them: fit each model per imputation, average
predictions at inference time. This is a **prediction-averaging /
ensembling** approach, not a Rubin's-rules inferential-pooling approach. If
you later want proper Rubin's-rules coefficient pooling specifically for
the OLS or logistic model (e.g. because you want a p-value on a
coefficient, not just a predictive score), that's a different, additional
piece of work this script doesn't attempt — flag it if you want it added.

---

## 5. Model registry

| Model key | Model | What it's doing here |
|---|---|---|
| `linear_regression` / `logistic_regression` | Ordinary least squares / logistic regression | Interpretable baseline — no regularization, no feature selection. If a more complex model doesn't beat this by much, the extra complexity probably isn't earning its keep. |
| `lasso` / `lasso_logistic` | L1-regularized (least absolute shrinkage) regression, via `LassoCV`/`LogisticRegressionCV(penalty="l1")` | Automatic feature selection via shrinkage — coefficients on unhelpful predictors get pushed to exactly zero. The `CV` variants auto-tune the regularization strength (`alpha`/`C`) via 5-fold cross-validation on the training set rather than using an arbitrary fixed value. |
| `svr_rbf` / `svc_rbf` | Support Vector Machine, RBF kernel | A nonlinear model that can capture interactions/curvature the linear models can't. Uses scikit-learn's default `C`/`gamma` — **not** hyperparameter-tuned (see §7). |
| `random_forest` | Random Forest (300 trees) | A second, different nonlinear baseline (tree-ensemble rather than kernel-based) — natively handles nonlinearity and interactions, less sensitive to feature scaling than SVM (scaling is applied anyway here for pipeline consistency, but doesn't materially change RF's behavior). |

Add or remove a model by editing `REGRESSION_MODELS`/`CLASSIFICATION_MODELS`
at the top of the file — nothing else in the script needs to change, since
`run_models_across_imputations` just iterates whatever's in the active
dict.

---

## 6. Metrics

| Task | Metric | Meaning |
|---|---|---|
| Regression | `r2` | Share of variance in the DV explained by the model (1.0 = perfect, 0.0 = no better than predicting the mean). |
| Regression | `rmse` | Root mean squared error, in the DV's own units — penalizes large errors more than small ones. |
| Regression | `mae` | Mean absolute error, in the DV's own units — more robust to outliers than RMSE. |
| Classification | `accuracy` | Share of test institutions correctly classified at a 0.5 probability threshold. |
| Classification | `roc_auc` | Probability the model ranks a random positive above a random negative — threshold-independent, usually the more informative number if your classes are imbalanced. |
| Classification | `f1` | Harmonic mean of precision and recall at the 0.5 threshold. |

---

## 7. Reading the output (`model_comparison.csv`)

One row per model, columns named `{ensembled|per_imputation}_{metric}[_mean|_std]`:

- **`ensembled_<metric>`** — score of the single prediction you'd actually
  deploy (the `M`-way average). This is the number to lead with when
  reporting results.
- **`per_imputation_<metric>_mean`** — average of the metric computed
  separately on each of the `M` fits. Usually close to the ensembled score,
  but not identical (averaging *predictions* then scoring isn't the same
  arithmetic operation as averaging *scores*).
- **`per_imputation_<metric>_std`** — how much that metric moved across the
  `M` individual fits. A small std means the model's apparent performance
  barely depends on which imputed dataset it happened to see; a large std
  means the model (or that metric) is sensitive to imputation noise, and
  the ensembled number is doing real work smoothing that out — worth
  flagging in any write-up rather than only reporting the single ensembled
  figure.

---

## 8. Known limitations / not built here

- **Single train/test split, not cross-validation.** Fine for a first
  "explore predictive power" pass; a k-fold CV nested inside each
  imputation would give more stable estimates but multiplies the fitting
  cost by the fold count on top of the `M` imputations, and wasn't asked
  for. Natural next step if you want tighter estimates later.
- **`svr_rbf`/`svc_rbf` and `random_forest` are not hyperparameter-tuned** —
  they run on scikit-learn's defaults (aside from `n_estimators=300` for
  the forest). Only the two Lasso variants auto-tune (via their built-in
  `CV` classes) because that came essentially free. If the SVM or Random
  Forest results look uncompetitive, that may be an untuned-hyperparameter
  artifact rather than a real statement about the model class — worth a
  `GridSearchCV`/`RandomizedSearchCV` pass before drawing conclusions from
  those two rows specifically.
- **No feature importance / coefficient inspection** — this script only
  reports accuracy-style metrics, not which features are driving them.
  Natural follow-on once you've picked a model worth interpreting.
- **No Rubin's-rules pooling option** for the linear/logistic models, even
  though it's the more standard approach for those two specifically — see
  §4.
- **MCMC / Bayesian treatment** — intentionally not started, per your note
  that it's later work.

---

## 9. Troubleshooting

- **`FileNotFoundError` on `completed_i.csv`** — `N_DATASETS` here doesn't
  match how many datasets `mice_pipeline.py` actually wrote; check
  `mice_output/` directly.
- **`KeyError` on `DV_COLUMN` or `UNITID`** — either `DV_COLUMN` is still
  the placeholder value, or your DV CSV doesn't have a `UNITID` column
  named exactly that (case-sensitive).
- **A large "Dropped N institutions with no `<DV_COLUMN>` value" message** —
  expected if your DV is only defined for a subset of institutions (e.g. an
  earnings outcome that's suppressed or unavailable for small/non-Title-IV
  schools); worth confirming that subset isn't systematically different
  from the full population before trusting the results as representative.
- **`FutureWarning`s from `LogisticRegressionCV`/`SVC`** — cosmetic,
  version-specific scikit-learn deprecation notices, not a correctness
  issue; safe to ignore unless you're on an unusually new scikit-learn and
  want to silence them explicitly.
