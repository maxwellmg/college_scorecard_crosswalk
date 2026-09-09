"""
Diagnostic instrumentation for the miceforest "OSError: access violation" crash.

Import this module BEFORE building/running your ImputationKernel — it
monkeypatches ImputationKernel._make_features_label, the exact miceforest
function (imputation_kernel.py line ~571) that builds each variable's
training X/y right before handing it to LightGBM's Dataset/train(). This is
purely observational: it calls the real method and prints/saves what comes
back, it doesn't change what gets trained or how.

Usage:
    import diagnose_mice_crash  # noqa: F401 — patches miceforest on import
    # ... then run mice_pipeline.py's main(), or call kernel.mice() directly, as normal

What this buys you that the crash traceback alone doesn't:

1. faulthandler.enable() — on some platforms this prints a raw C-level
   stack trace when a hard native crash happens, in addition to whatever
   OSError Python itself manages to raise. Free, can only add information,
   never hides anything the OSError already gave you.

2. A per-variable diagnostic line for every variable's label — dtype, row
   count, missing count, category codes / min-max / nunique — printed with
   flush=True so it survives even if the crash happens before Python's
   normal stdout buffering would have flushed it.

3. last_attempted_variable.pkl — overwritten before every training attempt,
   so after a crash it contains the EXACT features/label pair that was
   about to be trained. This is the most direct diagnostic available: load
   it after a crash and look at the real data yourself, rather than
   inferring from summary stats.

Why this matters specifically for this crash: miceforest's default
data_subset=0 means no internal subsampling happens — the label handed to
LightGBM should always be the FULL non-missing slice of that column, with
zero missing values, by construction. If the diagnostic below ever shows
n_missing > 0, or an unexpected dtype, for the variable that crashes, that
is the concrete, actionable finding — not a "some column is somehow rare"
guess.
"""

import faulthandler
import pickle
from pathlib import Path

from miceforest.imputation_kernel import ImputationKernel

DIAG_DIR = Path("mice_diagnostics")
DIAG_DIR.mkdir(exist_ok=True)

faulthandler.enable()

_original_make_features_label = ImputationKernel._make_features_label


def _diagnostic_make_features_label(self, variable, seed):
    features, label = _original_make_features_label(self, variable, seed)

    # Always overwritten — after a crash, whatever's on disk here is the
    # LAST variable that was attempted, i.e. very likely the one that crashed.
    with open(DIAG_DIR / "last_attempted_variable.pkl", "wb") as f:
        pickle.dump({"variable": variable, "features": features, "label": label}, f)

    if label.dtype.name == "category":
        codes = label.cat.codes
        print(
            f"[DIAG] {variable!r}: n={len(label)} dtype=category "
            f"n_missing={int(label.isna().sum())} n_neg1_codes={int((codes == -1).sum())} "
            f"categories_present={sorted(label.dropna().unique().tolist())} "
            f"value_counts={label.value_counts(dropna=False).to_dict()}",
            flush=True,
        )
    else:
        print(
            f"[DIAG] {variable!r}: n={len(label)} dtype={label.dtype} "
            f"n_missing={int(label.isna().sum())} nunique={int(label.nunique())} "
            f"min={label.min()} max={label.max()}",
            flush=True,
        )

    all_nan_feature_cols = features.columns[features.isna().all()].tolist()
    if all_nan_feature_cols:
        print(f"[DIAG]   WARNING — fully-NaN feature columns for {variable!r}: {all_nan_feature_cols}", flush=True)

    return features, label


ImputationKernel._make_features_label = _diagnostic_make_features_label
print("miceforest instrumented: per-variable diagnostics + faulthandler enabled.")
