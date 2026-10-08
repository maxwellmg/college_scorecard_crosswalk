"""
College Scorecard — MICE preprocessing + imputation pipeline
==============================================================
Companion code to README_MICE.md. Implements the domain-specific fixes
documented there (ordinal recoding, structural "-2"/"not applicable"
handling, RELAFFIL denomination collapsing, cardinality reduction) and then
runs multiple imputation with miceforest.

Designed to be run on whatever subset of columns survives your own
coverage-based pruning — every domain-specific step below is gated on
"if this column is still present," so dropping any number of variables
(a handful, or most of them) before or via COVERAGE_THRESHOLD never breaks
the script. Nothing hardcodes the full 3,308-column layout.

Also handles either of College Scorecard's two column-naming conventions:
the flat VARIABLE NAME convention (`PREDDEG`, `CONTROL`, `CCBASIC`, as in the
bulk "Most Recent Cohorts" CSV) or the dotted "developer-friendly name"
convention historical/API-sourced pulls use instead (`school.degrees_awarded.
predominant`, `school.ownership`, `school.carnegie_basic`). Every registry
and function below is written in terms of the flat convention; a dotted
input file gets renamed to match immediately after load (see
build_dotted_to_flat/rename_to_flat_convention) so nothing downstream needs
to know or care which convention the source file used.

Dependencies: pandas, numpy (present), plus `pip install miceforest` for the
actual imputation step (not installed in this environment — everything up to
and including column classification has been run against the real
Most-Recent-Cohorts-Institution.csv, and separately against a synthetic
dotted-convention sample, to confirm both naming conventions behave
correctly; the `run_mice()` call itself has not been executed here).
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────
# Config — tune these, nothing else below needs to change to match
# ────────────────────────────────────────────────────────────────────────

RAW_CSV = "Most-Recent-Cohorts-Institution.csv"

# Path to CollegeScorecardDataDictionary.csv — needed only to translate a
# dotted-convention input file to the flat convention (see module docstring).
# Point this at wherever your own copy lives; if RAW_CSV is already
# flat-named (like Most-Recent-Cohorts-Institution.csv), this file is read
# but the resulting rename map ends up empty — harmless either way.
DICTIONARY_CSV = "CollegeScorecardDataDictionary.csv"

OUTPUT_DIR = Path("mice_output")

# UNITID -> INSTNM lookup, built once and left alone after that. INSTNM
# (the actual school name) is deliberately excluded from the modeling
# pipeline itself (see ALWAYS_EXCLUDE) — a free-text institution name has
# no business being a MICE predictor or target. This table is the
# supported way to get names back for human review: merge it onto
# completed_{i}.csv on UNITID after MICE finishes. Kept separate from the
# modeling data on purpose, not merged in automatically here.
REFERENCE_TABLE_PATH = Path("unitid_instnm_reference.csv")

COVERAGE_THRESHOLD = 0.85   # the "magic proportion" — min non-missing share to keep a column
N_DATASETS = 5              # M — number of completed datasets MICE produces
N_ITERATIONS = 5            # MICE iterations per dataset
RANDOM_STATE = 42

# Memory lever, opt-in: run ONE of the M=N_DATASETS chains per process
# instead of all of them in one. See run_mice_one_dataset()'s docstring for
# the actual mechanism — miceforest holds every dataset's full working-data
# copy in memory simultaneously for the whole run regardless of which one
# is actively training, so num_datasets=5 costs roughly 5x a single
# dataset's peak memory in ONE process. Set this to an int (0 through
# N_DATASETS-1) and run this script that many times, each a genuine
# process restart (`python mice_pipeline.py 0`, then `1`, ...) — leave as
# None to run every dataset in a single process the original way, which is
# fine as long as that fits in memory.
DATASET_INDEX: int | None = None

# Whether MICE prints per-variable progress. Independent of DATASET_INDEX
# above — this affects a different resource than memory, specifically a
# notebook's rendered output size. kernel.mice(verbose=True) prints one
# line per variable per iteration per dataset — at ~270 columns x 5
# iterations x 5 datasets that's thousands of lines, which a Jupyter
# notebook's browser tab has to hold and render in its DOM as the cell's
# output grows. That accumulation is a well-known way to exhaust a
# browser tab's memory and is at least as likely an explanation for an
# "out of memory" error reported by a browser specifically as the Python
# process itself running out of RAM — the two have different fixes. Set to
# False once a run is stable and you no longer need the per-variable crash
# diagnostics verbose=True exists for (see run_mice()'s docstring on that).
MICE_VERBOSE = True

# Keep MICE_VERBOSE=True (so a stalled/crashed run still shows progress and
# roughly where it got to) while cutting the line volume that's the actual
# browser-memory concern above: only print every VERBOSE_PRINT_EVERY-th
# per-variable line. See _throttle_miceforest_verbose()'s docstring for the
# mechanism and why this doesn't cost any crash-diagnosis fidelity —
# find_crashed_variable() identifies the crashed variable from miceforest's
# internal logger state directly, not by parsing what got printed, so a
# crash on a variable that happened to be skipped here is still identified
# exactly as precisely as before. Set to 1 to print every variable (the
# original behavior).
VERBOSE_PRINT_EVERY = 50

# Speed lever, opt-in: by default, miceforest builds a separate LightGBM
# model per variable-with-missing-values using EVERY OTHER surviving column
# as a predictor (confirmed against miceforest's own source — see
# build_variable_schema()'s docstring). That's fine at ~270 columns; it
# stops being fine as column count grows, since per-variable training cost
# scales with predictor count too, not just row count. Set this to an int
# (e.g. 30) to cap each variable's predictor set to its N most-correlated
# other columns via build_variable_schema() — lets you raise
# COVERAGE_THRESHOLD/loosen the degenerate-column filters to keep MORE
# columns without training cost blowing up per-model. None preserves
# miceforest's default "everything predicts everything" behavior.
MAX_PREDICTORS_PER_VARIABLE: int | None = None

# LightGBM thread count for every model miceforest fits. -1 lets LightGBM
# use all available cores — the default; this was hardcoded to 1 earlier
# in this project specifically to rule out an OpenMP threading race as the
# cause of the "access violation" native crash documented in
# HANDOFF_MICE_CRASH_DEBUG.md. That crash's real causes turned out to be
# two confirmed data/library bugs unrelated to threading — the "PS" sentinel
# corrupting column classification (see load_raw()) and miceforest's
# logodds() divide-by-zero for lopsided binary columns (see run_mice()) —
# so single-threading is very likely no longer buying you anything but
# lost speed. If a fixed run is stable, there's no remaining reason found
# in this project's investigation to keep this at 1; revert to 1 only if
# you see the exact "access violation" crash recur after both of those
# fixes, as a bisection step.
LGB_NUM_THREADS = -1

# Set to False if you're pruning columns yourself upstream (e.g. in a
# spreadsheet) and want to hand this script an already-trimmed CSV —
# every other step still runs unconditionally on whatever columns arrive.
APPLY_COVERAGE_FILTER = True

# A categorical column whose 2nd-most-common realized category has fewer
# than this many rows gets dropped outright (see drop_degenerate_columns).
# This is what a LightGBM classifier target needs to not be a coin flip
# away from a class with ~0 training examples — which is what produces
# both the "very rare categories ... 0.0 probabilities" warning and, in the
# worst case, a hard crash (miceforest/LightGBM building a training split
# with 0-1 examples of a class can segfault rather than error cleanly,
# especially on Windows). Tune down if you'd rather keep more borderline
# columns; tune up if warnings/crashes persist.
#
# This is an absolute floor, which is easy to under-shoot on a smaller
# dataset: 20 rows is 0.5% of a 3,704-row panel but 0.02% of a 100,000-row
# one. MIN_CATEGORY_FRACTION sets a relative floor alongside it — the
# column is dropped if the 2nd-most-common category fails EITHER floor
# (whichever number is larger for your row count), so the check doesn't
# quietly loosen as the dataset gets smaller.
MIN_CATEGORY_COUNT = 20
MIN_CATEGORY_FRACTION = 0.01   # 1% of rows

# The continuous-column analog of MIN_CATEGORY_COUNT/FRACTION: a numeric
# column where one single value accounts for at least this share of its
# non-missing rows is dropped (see drop_near_constant_continuous). A
# classification target with a near-empty class is the failure mode the
# nominal/binary check above exists for; a regression target that's
# essentially a constant (near-zero variance) is the analogous risk on the
# continuous side — not guaranteed to crash the same way, but degenerate
# enough that it isn't contributing real signal either, and it's cheap
# insurance against whatever the exact failure condition turns out to be.
MAX_DOMINANT_VALUE_SHARE = 0.99

# Columns whose name ends with one of these (case-insensitive) are
# per-variable provenance metadata — e.g. a "<variable>.year_added" column
# recording which year THAT variable was introduced, not an institutional
# measurement itself. Real and worth keeping, but has no business being
# imputed or used as a MICE predictor: it describes the data, not the
# institution. split_metadata_columns() sets these aside into their own
# frame rather than dropping them.
METADATA_COLUMN_SUFFIXES = ("year_added",)

# Registered rare-but-meaningful binary flags exempt from that auto-drop —
# these are expected to be lopsided (e.g. ~100 HBCUs nationally) and dropping
# them isn't a "make MICE stable" call, it's a "delete real information"
# call. Unregistered columns (anything not in this doc's domain registries —
# which is most of a 2,000+ column extract) get no such protection and are
# dropped if they trip MIN_CATEGORY_COUNT, since there's no domain basis
# here to judge whether their rarity is meaningful or just noise.
PROTECTED_FROM_VARIANCE_DROP = {
    "HBCU", "PBI", "ANNHI", "TRIBAL", "AANAPII", "HSI", "NANTI",
    "MENONLY", "WOMENONLY",
}

# ────────────────────────────────────────────────────────────────────────
# Domain registries — from CollegeScorecardDataDictionary.csv + verification
# against the real extract (see README_MICE.md). Only ever consulted via
# "if this column is present" checks, so pruning any of these upstream is safe.
# ────────────────────────────────────────────────────────────────────────

# Pure identifiers / free text / redundant-with-something-kept fields.
# Dropped unconditionally, before coverage is even computed — never useful
# as MICE predictors or targets regardless of how much data they have.
# NOTE: UNITID is NOT here — see load_raw(), which pulls it out as the
# DataFrame index instead of dropping it, so it survives as a join key on
# the other side of MICE (e.g. to attach a dependent variable afterward).
ALWAYS_EXCLUDE = {
    "OPEID", "OPEID6", "INSTNM", "CITY", "STABBR", "ZIP",
    "INSTURL", "NPCURL", "ACCREDCODE",
    "ST_FIPS",           # redundant with REGION, much higher cardinality
    "SCORECARD_SECTOR",  # deterministic function of CONTROL x PREDDEG
    "LOCALE2",           # verified 100% missing in the current extract
}

# var -> raw code meaning "structurally not applicable" (not missing).
# Split into a boolean "<var>_APPLICABLE" flag; the sentinel itself is
# excluded from the substantive scale rather than imputed.
STRUCTURAL_NA = {
    "CCBASIC": -2,
    "CCUGPROF": -2,
    "CCSIZSET": -2,
    "OPENADMP": 3,   # "does not enroll first-time students"
}

# var -> ordered list of raw codes, low -> high, OR None meaning "ascending
# numeric sort of the observed codes already matches the intended order."
# Codes present in ORDINAL_MISSING_CODES[var] are treated as missing, not a
# level, before ranking.
ORDINAL_ORDER: dict[str, list | None] = {
    "PREDDEG": None,
    "HIGHDEG": None,
    "ICLEVEL": None,
    "LOCALE": None,
    "ADMCON7": [3, 5, 2, 1],   # neither < considered-not-required < recommended < required
}
ORDINAL_MISSING_CODES = {
    "ADMCON7": {4},   # "do not know" is not a stringency level
}

# Already single 0/1 columns in the raw file (verified) — never one-hot
# these even if a generic categorical-encoding step runs elsewhere.
BINARY_COLS = {
    "MAIN", "HBCU", "PBI", "ANNHI", "TRIBAL", "AANAPII", "HSI", "NANTI",
    "MENONLY", "WOMENONLY", "DISTANCEONLY", "CURROPER", "DOLPROVIDER",
}

# Integer-coded nominal variables. These MUST be registered explicitly —
# unlike ACCREDAGENCY or a post-collapse RELAFFIL, they're small-integer
# columns indistinguishable from a continuous numeric column by dtype alone,
# so the dtype-based fallback in classify_columns() would otherwise treat
# them as continuous and hand them to MICE as if ordered.
NOMINAL_COLS = {"CONTROL", "REGION", "SCHTYPE", "OPEFLAG"}

# These are also exempt from drop_degenerate_columns, for the same reason
# as the minority-serving flags above: REGION's "U.S. Service Schools" or
# OPEFLAG's less common Title IV statuses are rare nationally but real,
# curated categories, not noise — dropping the whole variable over one rare
# level throws away the other 9 (or however many) good categories to "fix"
# one, which is a worse trade than just leaving it be.
PROTECTED_FROM_VARIANCE_DROP |= NOMINAL_COLS

CIP_COLS = [f"CIPCODE{i}" for i in range(1, 7)]

# RELAFFIL: ~80 denominations collapsed to families. Not exhaustive by
# construction — anything observed but not listed here falls to "Other"
# rather than becoming spuriously missing (see collapse_relaffil). Review/
# edit this mapping if the family boundaries matter to the analysis; it's a
# judgment call, not a value from the dictionary.
RELAFFIL_FAMILY_MAP = {
    30: "Catholic",
    80: "Jewish",
    106: "Muslim",
    94: "Latter Day Saints",
    91: "Orthodox Christian", 92: "Orthodox Christian", 110: "Orthodox Christian",
    93: "Other", 65: "Other",
    42: "Interdenominational", 78: "Interdenominational", 108: "Non-Denominational", 88: "Non-Denominational",
    # Mainline Protestant
    22: "Mainline Protestant", 39: "Mainline Protestant", 53: "Mainline Protestant",
    66: "Mainline Protestant", 67: "Mainline Protestant", 68: "Mainline Protestant",
    71: "Mainline Protestant", 73: "Mainline Protestant", 76: "Mainline Protestant",
    50: "Mainline Protestant", 60: "Mainline Protestant", 61: "Mainline Protestant",
    97: "Mainline Protestant", 103: "Mainline Protestant",
    # Evangelical / other Protestant
    24: "Evangelical Protestant", 27: "Evangelical Protestant", 28: "Evangelical Protestant",
    33: "Evangelical Protestant", 34: "Evangelical Protestant", 35: "Evangelical Protestant",
    36: "Evangelical Protestant", 37: "Evangelical Protestant", 38: "Evangelical Protestant",
    40: "Evangelical Protestant", 41: "Evangelical Protestant", 43: "Evangelical Protestant",
    44: "Evangelical Protestant", 45: "Evangelical Protestant", 47: "Evangelical Protestant",
    48: "Evangelical Protestant", 49: "Evangelical Protestant", 51: "Evangelical Protestant",
    52: "Evangelical Protestant", 54: "Evangelical Protestant", 55: "Evangelical Protestant",
    57: "Evangelical Protestant", 58: "Evangelical Protestant", 59: "Evangelical Protestant",
    64: "Evangelical Protestant", 69: "Evangelical Protestant", 74: "Evangelical Protestant",
    75: "Evangelical Protestant", 77: "Evangelical Protestant", 79: "Evangelical Protestant",
    81: "Evangelical Protestant", 84: "Evangelical Protestant", 87: "Evangelical Protestant",
    89: "Evangelical Protestant", 95: "Evangelical Protestant", 99: "Evangelical Protestant",
    100: "Evangelical Protestant", 101: "Evangelical Protestant", 102: "Evangelical Protestant",
    105: "Evangelical Protestant", 107: "Evangelical Protestant",
}


# ────────────────────────────────────────────────────────────────────────
# Step 0 — naming convention crosswalk (dotted API names <-> flat VARIABLE NAME)
# ────────────────────────────────────────────────────────────────────────

def build_dotted_to_flat(dictionary_csv: str | Path) -> dict[str, str]:
    """CollegeScorecardDataDictionary.csv ties the flat VARIABLE NAME
    (`PREDDEG`) to the dev-category + developer-friendly name pair the API
    uses instead (`school` + `degrees_awarded.predominant` ->
    `school.degrees_awarded.predominant`). A `root`-category field has no
    prefix at all (`UNITID` -> `id`, not `root.id`) — matches how
    crosswalk.py's own fetch_scorecard() requests fields like `school.name`
    and bare `id` from the live API. Returns {dotted_name: flat_name} for
    every VARIABLE NAME row that has a developer-friendly name on file.
    """
    dotted_to_flat: dict[str, str] = {}
    seen: set[str] = set()
    with open(dictionary_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            var = row["VARIABLE NAME"].strip()
            if not var or var in seen:
                continue
            seen.add(var)
            friendly = row["developer-friendly name"].strip()
            if not friendly:
                continue
            category = row["dev-category"].strip()
            dotted = friendly if category in ("", "root") else f"{category}.{friendly}"
            dotted_to_flat[dotted] = var
    return dotted_to_flat


def rename_to_flat_convention(df: pd.DataFrame, dotted_to_flat: dict[str, str]) -> tuple[pd.DataFrame, int]:
    """No-op on a file that's already flat-named (e.g.
    Most-Recent-Cohorts-Institution.csv) — none of its columns match a
    dotted name, so the rename map ends up empty. On a dotted/API-style
    historical pull, renames every column the dictionary recognizes;
    anything it doesn't recognize passes through untouched and falls to
    classify_columns' generic dtype-based fallback like any other
    unregistered column — a partial dictionary match degrades gracefully
    rather than breaking the run."""
    rename_map = {c: dotted_to_flat[c] for c in df.columns if c in dotted_to_flat}
    return df.rename(columns=rename_map), len(rename_map)


# ────────────────────────────────────────────────────────────────────────
# Step 1 — load
# ────────────────────────────────────────────────────────────────────────

def load_raw(path: str | Path, dictionary_csv: str | Path | None = None) -> pd.DataFrame:
    """Load the extract, normalizing every known missing-value convention
    to real NaN at read time. `NA` is caught by pandas' default na_values.
    `PrivacySuppressed`/`NULL` are defensive (neither occurs in the flat
    extract this was first verified against) but `PS` is NOT defensive —
    it's a real, previously-missed suppression code used across 2,341 of
    this file's 3,308 columns (8.1M occurrences), mostly in the *_YR2_RT-
    style cohort completion/transfer/withdrawal rate families, and in the
    cohort-count/earnings families (`OVERALL_YR2_N`/`_YR3_N`/`_YR4_N`,
    `MD_EARN_WNE_4YR`, `COUNT_WNE_4YR`, ...).

    THIS IS NOT JUST A COVERAGE-ACCURACY BUG — it is very likely a direct
    contributor to the miceforest/LightGBM native crashes documented in
    HANDOFF_MICE_CRASH_DEBUG.md, and should be re-tested against that crash
    before assuming this project needs the R port
    (college_scorecard_crosswalk/mice_pipeline.R) instead of this file.
    Mechanism, confirmed against the real data:

    1. Without this fix, a column like MD_EARN_WNE_4YR (raw dollar
       earnings) reads as pandas dtype `object`/`str`, not numeric, because
       "PS" mixed in with numbers defeats pandas' automatic type inference
       for the whole column — `pd.api.types.is_numeric_dtype()` returns
       False on it.
    2. classify_columns()'s fallback (numeric -> continuous, else ->
       nominal) therefore misroutes it to NOMINAL, not continuous — a
       purely mechanical dtype check, with no way to know the column is
       "really" a suppressed-but-otherwise-continuous rate/dollar field.
    3. finalize_dtypes() then runs it through reduce_cardinality(), which
       keeps only categories with >= 1% frequency. Confirmed against real
       data: for MD_EARN_WNE_4YR, "PS" appears 469 times (7.5% of rows,
       comfortably clears the threshold) while every actual dollar amount
       appears at most 31 times (0.5%, below it) — so EVERY real value
       collapses into a single "Other" bucket. A genuinely continuous,
       information-rich earnings variable becomes a fake 2-level factor:
       "PS" vs "Other".
    4. That fake factor then gets handed to LightGBM as a classification
       target (when it's this variable's turn to be imputed) or as a
       `categorical_feature` predictor (for every other variable's turn).
       Because privacy suppression is driven by small cohort size, "PS"
       fires together across many related cohort/earnings columns for the
       same (small) institutions at once — so this bug doesn't corrupt one
       column in isolation, it manufactures a whole cluster of near-
       identical, spuriously-binary "PS-vs-everything" columns that are
       highly redundant with each other. That is exactly the kind of
       degenerate, redundant categorical input this project's crash
       debugging (drop_degenerate_columns, drop_near_constant_continuous,
       the retry-with-removal loop) was built to catch and never fully
       could — because those filters run on the ALREADY-mangled 2-level
       version of the column, which looks individually well-populated
       (469 vs ~5,200 rows) and doesn't trip a rarity threshold at all. The
       real defect was upstream, in what the column WAS before it reached
       those filters, not in the filters' thresholds.

    Fixing it here, at load time, is what makes columns like
    MD_EARN_WNE_4YR flow through the rest of this pipeline as the
    continuous variables they actually are — never becoming a nominal
    column, never going through reduce_cardinality, never becoming a
    LightGBM classification target/categorical predictor at all for this
    reason. Whether this alone resolves the native crash can only be
    confirmed by re-running run_mice_with_retry() against real data; it
    removes a definitively confirmed, severe data-corruption bug that
    affects roughly 70% of this file's columns, which is a stronger and
    more specific candidate than anything else investigated in
    HANDOFF_MICE_CRASH_DEBUG.md (a broken LightGBM install, a corrupted
    process) — but it is a strong hypothesis backed by direct evidence, not
    a proven fix, since the crash itself could only ever be reproduced on
    the original Windows environment, not here.

    Separately, this also means every coverage/sparsity number reported by
    this script in any session before this fix was silently wrong for the
    ~2,341 affected columns — "PS" was being counted as a present value at
    every coverage-filtering decision point until this fix moved the
    normalization to load time instead of a later
    pd.to_numeric(errors="coerce") cleanup.

    dictionary_csv defaults to None (resolved to the module-level
    DICTIONARY_CSV below) rather than `= DICTIONARY_CSV` directly in the
    signature: a plain default like that is evaluated once, when this
    function is first defined, not each time it's called — so editing
    DICTIONARY_CSV afterward (e.g. in a Jupyter/Spyder session where this
    module was already imported) would silently have no effect on calls
    that rely on the default. Resolving it inside the function body means
    the current value of the module-level constant is always what's used.
    """
    if dictionary_csv is None:
        dictionary_csv = DICTIONARY_CSV
    df = pd.read_csv(
        path,
        na_values=["PrivacySuppressed", "NULL", "PS"],
        keep_default_na=True,
        low_memory=False,
    )
    dotted_to_flat = build_dotted_to_flat(dictionary_csv)
    df, n_renamed = rename_to_flat_convention(df, dotted_to_flat)
    print(f"Renamed {n_renamed} dotted-convention column(s) to the flat convention "
          f"(0 is expected/harmless if the input file is already flat-named).")
    if "UNITID" in df.columns:
        df = df.set_index("UNITID")
    df = df.drop(columns=[c for c in ALWAYS_EXCLUDE if c in df.columns])
    return df


# ────────────────────────────────────────────────────────────────────────
# Step 1b — set aside per-variable metadata columns (e.g. "*_year_added"),
# then checkpoint the sparsity of whatever's left before MICE sees any of it
# ────────────────────────────────────────────────────────────────────────

def split_metadata_columns(
    df: pd.DataFrame, suffixes: tuple[str, ...] = METADATA_COLUMN_SUFFIXES
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Pulls out columns matching METADATA_COLUMN_SUFFIXES into their own
    frame (same UNITID index as the input, so it can be rejoined later)
    rather than dropping them — they're real, worth keeping, just not
    modeling variables. Returns (mice_df, metadata_df, metadata_column_names)
    so what got set aside is visible rather than silent. Matching is a
    case-insensitive suffix check, so this catches the suffix regardless of
    whether the column is in the flat or dotted naming convention.
    """
    suffixes_lower = tuple(s.lower() for s in suffixes)
    metadata_cols = [c for c in df.columns if c.lower().endswith(suffixes_lower)]
    return df.drop(columns=metadata_cols), df[metadata_cols].copy(), metadata_cols


def report_sparsity(df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Coverage (share non-missing) for every column, worst-first. Meant to
    be run as a checkpoint right after split_metadata_columns and
    apply_structural_fixes but BEFORE select_by_coverage actually filters
    anything — so the full distribution is visible and COVERAGE_THRESHOLD
    can be sanity-checked against it, rather than only ever seeing
    pass/fail counts after the fact.
    """
    coverage = df.notna().mean().sort_values()
    coverage.name = "coverage"
    out = coverage.reset_index().rename(columns={"index": "column"})
    return out.head(top_n) if top_n is not None else out


# ────────────────────────────────────────────────────────────────────────
# Step 2 — domain-specific structural fixes (run BEFORE coverage filtering,
# so coverage reflects the corrected missingness, not the raw column's)
# ────────────────────────────────────────────────────────────────────────

def apply_structural_na(df: pd.DataFrame) -> pd.DataFrame:
    """Split each STRUCTURAL_NA variable into an '<var>_APPLICABLE' flag
    plus a substantive column with the sentinel removed. A row that was
    already missing stays missing on the flag too (unknown applicability),
    rather than defaulting to True or False."""
    df = df.copy()
    for var, sentinel in STRUCTURAL_NA.items():
        if var not in df.columns:
            continue
        is_missing = df[var].isna()
        applicable = np.where(is_missing, np.nan, (df[var] != sentinel).astype(float))
        df[f"{var}_APPLICABLE"] = applicable
        df.loc[df[var] == sentinel, var] = np.nan
    return df


def recode_openadmp(df: pd.DataFrame) -> pd.DataFrame:
    """After apply_structural_na strips out code 3, OPENADMP is left with
    only {1=Yes, 2=No}. Recode to a plain 0/1 so it's treated as binary
    downstream instead of carrying its original 1/2 coding."""
    if "OPENADMP" in df.columns:
        df = df.copy()
        df["OPENADMP"] = df["OPENADMP"].map({1: 1, 2: 0})
    return df


def collapse_relaffil(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse RELAFFIL's ~80 codes to denomination families. Codes present
    in the data but absent from RELAFFIL_FAMILY_MAP become 'Other' rather
    than NaN — only genuinely missing cells stay missing."""
    if "RELAFFIL" not in df.columns:
        return df
    df = df.copy()
    mapped = df["RELAFFIL"].map(RELAFFIL_FAMILY_MAP)
    unmapped_but_present = mapped.isna() & df["RELAFFIL"].notna()
    mapped = mapped.mask(unmapped_but_present, "Other")
    df["RELAFFIL"] = mapped
    return df


def encode_ordinal(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Recode each ORDINAL_ORDER variable present in df to a dense integer
    rank (0-indexed), respecting ORDINAL_MISSING_CODES. Kept as plain
    numeric (not pandas 'category') on purpose — see the module docstring
    in run_mice() for why. Returns the rank maps too, needed later to snap
    imputed values back to valid categories and to decode results."""
    df = df.copy()
    rank_maps: dict[str, dict] = {}
    for var, order in ORDINAL_ORDER.items():
        if var not in df.columns:
            continue
        missing_codes = ORDINAL_MISSING_CODES.get(var, set())
        working = df[var].where(~df[var].isin(missing_codes))
        categories = order if order is not None else sorted(working.dropna().unique())
        rank_map = {cat: rank for rank, cat in enumerate(categories)}
        df[var] = working.map(rank_map)
        rank_maps[var] = rank_map
    return df, rank_maps


def prepare_cip_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Roll 6-digit CIP program codes up to their 2-digit family before
    cardinality reduction — the 6-digit level is too granular to bucket
    sensibly otherwise."""
    df = df.copy()
    for col in CIP_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("string").str.slice(0, 2)
    return df


def reduce_cardinality(series: pd.Series, max_levels: int = 15, min_freq: float = 0.01) -> pd.Series:
    """Generic fallback for any nominal column — registered or not — whose
    raw cardinality actually exceeds max_levels (e.g. ACCREDAGENCY, RELAFFIL,
    a rolled-up CIPCODE). Left alone otherwise: gating on total category
    count first, not just per-category frequency, means a naturally small
    nominal like REGION or OPEFLAG never gets touched even though some of
    its categories are individually rare (e.g. REGION's "U.S. Service
    Schools") — those are real, meaningful categories, not noise from an
    oversized cardinality. Missing values are left as NaN, not folded into
    'Other'.

    Gating on cardinality here assumes the column got here LEGITIMATELY
    nominal — i.e. that classify_columns() didn't misroute a genuinely
    continuous column into this path because an un-normalized text sentinel
    (see load_raw()'s "PS" docstring) defeated its numeric-dtype check. For
    a column like that, this function is precisely the mechanism that
    destroys it: near-unique real values (each individually below
    min_freq) all collapse into a single "Other" bucket, leaving a fake
    2-or-3-level factor built entirely out of missingness structure rather
    than the variable's actual meaning. Confirmed against real Scorecard
    data — fix the sentinel at load time, not here."""
    counts = series.value_counts(dropna=True)
    if counts.shape[0] <= max_levels:
        return series
    share = counts / counts.sum()
    keep = set(share[share >= min_freq].index[:max_levels])
    return series.where(series.isna() | series.isin(keep), other="Other")


def apply_structural_fixes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Runs every domain-specific fix, each gated on column presence.
    Safe to call on a dataframe that's missing any number of these columns."""
    df = apply_structural_na(df)
    df = recode_openadmp(df)
    df = collapse_relaffil(df)
    df = prepare_cip_cols(df)
    df, rank_maps = encode_ordinal(df)
    return df, rank_maps


# ────────────────────────────────────────────────────────────────────────
# Step 3 — coverage-based column selection ("the magic proportion")
# ────────────────────────────────────────────────────────────────────────

def drop_degenerate_columns(
    df: pd.DataFrame,
    columns_to_check: list[str],
    min_category_count: int = MIN_CATEGORY_COUNT,
    min_category_fraction: float = MIN_CATEGORY_FRACTION,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop any column in columns_to_check whose SMALLEST realized category
    fails EITHER the absolute or the relative-to-row-count floor — the
    condition that produces miceforest's "very rare categories" warning
    and, in the worst case, a LightGBM training crash on a near-empty
    class. Columns in PROTECTED_FROM_VARIANCE_DROP are exempt (see its
    docstring).

    columns_to_check should be classify_columns()'s "nominal" + "binary"
    lists ONLY — this check exists because a classification target with a
    near-empty class can crash LightGBM's classifier training. Ordinal and
    continuous columns are handled as regression targets instead (see
    encode_ordinal's docstring for why ordinal is kept numeric rather than
    categorical), and a regression target has no analogous "empty class"
    failure mode — a rare value there is just a slightly unusual number,
    not a degenerate training split. Running this check against ALL columns
    indiscriminately (an earlier version of this function did, using an
    nunique-based heuristic to guess which were categorical) will catch
    small-cardinality ORDINAL columns too and drop a real, safe-to-keep
    feature for no actual safety benefit — hence taking an explicit column
    list instead of guessing from dtype/cardinality.

    Checks the smallest category (`counts.iloc[-1]`), not the 2nd-largest:
    for a binary column those are the same value, but a nominal column with
    3+ categories could have two comfortably-sized categories and one
    near-empty one — checking only the runner-up would miss exactly the
    category causing the problem.
    """
    required = max(min_category_count, min_category_fraction * len(df))
    dropped = []
    for col in columns_to_check:
        if col in PROTECTED_FROM_VARIANCE_DROP:
            continue
        counts = df[col].value_counts(dropna=True)
        if counts.shape[0] < 2:
            dropped.append(col)  # constant or entirely missing — useless either way
            continue
        if counts.iloc[-1] < required:
            dropped.append(col)
    return df.drop(columns=dropped), dropped


def report_category_rarity(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Diagnostic only — not called anywhere in the main pipeline. Run this
    yourself on the frame build_analysis_frame() returns (after
    drop_degenerate_columns and finalize_dtypes have already run) to see
    which SURVIVING nominal/binary columns are closest to the edge, sorted
    worst-first. This is the fast (pure pandas, no LightGBM) way to check
    "did tightening the threshold actually help, and by how much" without
    spending 20 minutes on a real MICE attempt each time you adjust
    MIN_CATEGORY_COUNT/MIN_CATEGORY_FRACTION.

    Checks dtype == 'category' rather than re-guessing from cardinality —
    by the time build_analysis_frame() has returned, finalize_dtypes has
    already cast exactly the nominal/binary columns to pandas 'category',
    so this can just ask dtype directly instead of approximating it.

    Example:
        df, report, rank_maps = build_analysis_frame(RAW_CSV, DICTIONARY_CSV)
        print(report_category_rarity(df))
    """
    rows = []
    for col in df.columns:
        series = df[col]
        if not isinstance(series.dtype, pd.CategoricalDtype):
            continue
        counts = series.value_counts(dropna=True)
        if counts.shape[0] < 2:
            continue
        rows.append({
            "column": col,
            "smallest_category_count": counts.iloc[-1],
            "smallest_category_share": counts.iloc[-1] / counts.sum(),
            "n_categories": counts.shape[0],
        })
    return pd.DataFrame(rows).sort_values("smallest_category_count").head(top_n)


def drop_near_constant_continuous(
    df: pd.DataFrame,
    continuous_cols: list[str],
    max_dominant_share: float = MAX_DOMINANT_VALUE_SHARE,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop any continuous column where a single value accounts for at
    least max_dominant_share of its non-missing rows — the regression-target
    analog of drop_degenerate_columns (see MAX_DOMINANT_VALUE_SHARE's
    comment for why). Only ever called on columns already classified as
    continuous — an ordinal column with a dominant rank is fine (it's
    imputed via the same regression-style path deliberately, and a common
    rank isn't a training hazard the way a near-empty class is).
    """
    required = max_dominant_share
    dropped = []
    for col in continuous_cols:
        counts = df[col].value_counts(dropna=True, normalize=True)
        if counts.empty:
            dropped.append(col)  # entirely missing
            continue
        if counts.iloc[0] >= required:
            dropped.append(col)
    return df.drop(columns=dropped), dropped


def select_by_coverage(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    """Keep only columns with >= threshold share of non-missing values,
    measured AFTER apply_structural_fixes (so a Carnegie field's coverage
    reflects real missingness, not inflated by counting '-2' as present).
    Returns the trimmed frame and a coverage report for every dropped
    column, sorted worst-first, so the exclusion list is auditable."""
    coverage = df.notna().mean().sort_values()
    dropped = coverage[coverage < threshold]
    kept_cols = coverage[coverage >= threshold].index.tolist()
    return df[kept_cols].copy(), dropped


# ────────────────────────────────────────────────────────────────────────
# Step 4 — classify whatever columns survived, dynamically
# ────────────────────────────────────────────────────────────────────────

def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Never assumes a fixed column set: registered ordinal/binary/nominal
    columns are claimed first (only if actually present), then every
    remaining column is classified by whether pandas considers it numeric.
    This is what makes the pipeline indifferent to how many variables got
    pruned upstream.

    Deciding the fallback by "is this numeric?" (continuous) rather than
    "is this string-typed?" (nominal) is deliberate: pandas' text-dtype name
    has changed across versions (object / StringDtype / pandas-3.x's default
    `str` dtype), so checking for numeric-ness and treating everything else
    as nominal is the version-stable way to catch a text column like
    ACCREDAGENCY without needing to enumerate every dtype name pandas might
    use for text.

    This numeric-dtype check is also why load_raw() normalizing every
    missing-value sentinel (see its docstring on the "PS" fix specifically)
    matters so much: a genuinely continuous column that still has an
    un-normalized text sentinel mixed into it reads as non-numeric here,
    gets classified nominal, and then gets its information destroyed by
    reduce_cardinality() below — not a hypothetical, confirmed against real
    data for dozens of Scorecard's earnings/cohort-count columns.
    """
    ordinal_cols = [c for c in ORDINAL_ORDER if c in df.columns]
    applicable_flags = [f"{v}_APPLICABLE" for v in STRUCTURAL_NA if f"{v}_APPLICABLE" in df.columns]
    binary_cols = [c for c in BINARY_COLS if c in df.columns]
    if "OPENADMP" in df.columns:
        binary_cols.append("OPENADMP")
    binary_cols += applicable_flags
    nominal_cols = [c for c in NOMINAL_COLS if c in df.columns]

    claimed = set(ordinal_cols) | set(binary_cols) | set(nominal_cols)
    remaining = [c for c in df.columns if c not in claimed]

    numeric_remaining = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]

    # A numeric column with exactly 2 realized values is a binary flag
    # regardless of whether it's been registered by name — e.g. the ~180
    # CIP##ASSOC/BACHL/CERT# program-offered flags are plain 0/1 integers
    # with no individual entry in BINARY_COLS. Left as "continuous" (the
    # naive numeric-dtype fallback), they'd be handed to MICE as regression
    # targets AND skip drop_degenerate_columns entirely (which only checks
    # the nominal+binary lists) — meaning a flag where only a handful of
    # institutions offer some narrow program would sail through with zero
    # rare-category protection. This check is deliberately narrow (exactly
    # 2 values, not "few values") — a genuine small-integer count field
    # (e.g. NUMBRANCH) can legitimately have low cardinality without being
    # conceptually binary, and reclassifying those as nominal/binary would
    # be a much less certain call than this one.
    newly_binary = [c for c in numeric_remaining if df[c].nunique(dropna=True) == 2]
    binary_cols += newly_binary

    continuous_cols = [c for c in numeric_remaining if c not in newly_binary]
    nominal_cols += [c for c in remaining if c not in numeric_remaining]

    return {
        "ordinal": ordinal_cols,
        "binary": binary_cols,
        "nominal": nominal_cols,
        "continuous": continuous_cols,
    }


def finalize_dtypes(df: pd.DataFrame, columns: dict[str, list[str]]) -> pd.DataFrame:
    """Reduce cardinality on nominal columns, then set final dtypes:
    numeric (float) for ordinal/continuous, pandas 'category' for
    binary/nominal — the dtype miceforest/LightGBM use to decide
    regression vs. classification per column."""
    df = df.copy()
    for c in columns["nominal"]:
        df[c] = reduce_cardinality(df[c])
    for c in columns["ordinal"] + columns["continuous"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in columns["binary"] + columns["nominal"]:
        df[c] = df[c].astype("category")
    return df


# ────────────────────────────────────────────────────────────────────────
# Step 5 — run MICE (miceforest)
# ────────────────────────────────────────────────────────────────────────

def _throttle_miceforest_verbose(every: int) -> None:
    """Patches miceforest's Logger.log so per-variable progress lines print
    only every `every`-th occurrence, while every OTHER logged line (the
    "N Dataset N" headers, which are low-volume and worth always seeing)
    prints normally. Exists because kernel.mice(verbose=True) is all-or-
    nothing — miceforest's own Logger.log() (miceforest/logger.py) is just
    `print(*args, **kwargs)` behind an `if self.verbose` check, with no
    sampling option — and per-variable lines are specifically what makes
    verbose output expensive: one per variable per iteration per dataset,
    thousands of lines on a run this size (see MICE_VERBOSE's docstring on
    why that volume is a plausible contributor to a browser-reported "out
    of memory" on its own, independent of the Python process's own RAM).

    Detects a per-variable line by its exact format from miceforest's own
    source (imputation_kernel.py: `logger.log(" | " + variable, end="")`)
    — anything else (iteration numbers, "Dataset N", the trailing newline)
    passes through untouched regardless of the counter. every=1 disables
    throttling (every line prints, the original behavior).

    Does not affect find_crashed_variable()'s accuracy — that function
    reads logger.started_timers directly (miceforest's own record of which
    (dataset, iteration, variable) started training but hasn't finished),
    not the printed text, so a crash on a variable whose line was skipped
    here is identified exactly as precisely as if every line had printed.

    Idempotent against repeated calls (e.g. re-running a Jupyter cell after
    an edit) — same marker-attribute guard as diagnose_mice_crash.py, for
    the same reason: without it, a second call would wrap an already-
    patched method again rather than replacing it cleanly, and the counter
    would reset while the double-wrapping compounded the throttle rate.
    """
    from miceforest.logger import Logger

    already_patched = getattr(Logger.log, "_is_throttle_patch", False)
    original_log = Logger.log.__wrapped__ if already_patched else Logger.log

    counter = {"n": 0}

    def throttled_log(self, *args, **kwargs):
        text = args[0] if args else ""
        if isinstance(text, str) and text.startswith(" | "):
            counter["n"] += 1
            if counter["n"] % every != 0:
                return
        original_log(self, *args, **kwargs)

    throttled_log._is_throttle_patch = True
    throttled_log.__wrapped__ = original_log
    Logger.log = throttled_log


def build_variable_schema(df: pd.DataFrame, max_predictors: int | None) -> dict[str, list[str]] | None:
    """Caps each variable-with-missing-values' predictor set to its
    max_predictors most-correlated other columns, instead of miceforest's
    default of every other column. Returns None (meaning "use miceforest's
    default schema") if max_predictors is None — this function is a no-op
    unless you've explicitly opted in via MAX_PREDICTORS_PER_VARIABLE.

    Why this is a speed lever and not a correctness one, unlike the
    equivalent step in mice_pipeline.R: LightGBM is tree-based — it doesn't
    invert any matrix that a collinear or oversized predictor set could
    make singular, so miceforest was never at risk of R's "computationally
    singular" failure from a wide predictor set. What a wide predictor set
    DOES cost here is pure training time: every one of the ~270+ variables'
    LightGBM models has to consider every other surviving column as a
    candidate split feature, for every one of N_ITERATIONS x N_DATASETS
    fits. That cost scales with column count regardless of whether the
    extra columns are actually informative for a given target — capping to
    the columns most correlated with each specific target keeps each
    individual model cheap even as you raise COVERAGE_THRESHOLD or loosen
    the degenerate-column filters to keep more columns overall.

    Only variables with actual missing values get a schema entry — matches
    miceforest's own default behavior (confirmed against its source:
    variable_schema=None trains models only for
    ImputedData.vars_with_any_missing, using every other column as
    predictors for each). Passing a schema that included fully-observed
    columns as targets would change what gets modeled, not just how.
    """
    if max_predictors is None:
        return None

    targets = df.columns[df.isna().any()].tolist()

    # Category dtype -> integer codes as a numeric stand-in for ranking
    # correlation strength. This is a coarser signal than a true
    # association measure for nominal columns (integer-coding an unordered
    # category imposes an arbitrary order), but it only has to rank
    # candidate predictors relative to each other for a speed cap, not
    # produce a statistically meaningful correlation — good enough for
    # that, and it's the same pragmatic approach used in mice_pipeline.R.
    # .cat.codes uses -1 for NaN (not NaN itself) — replace it back so the
    # fill step below doesn't treat "missing" as a real, extreme category.
    numeric_proxy = pd.DataFrame({
        col: (df[col].cat.codes.replace(-1, np.nan) if df[col].dtype.name == "category" else df[col])
        for col in df.columns
    })

    # DANGER, confirmed against real timing data — do not change this back
    # to df.corr(): pandas' DataFrame.corr() does NOT use a vectorized BLAS
    # path once any NaN is present (which is every column here) — it falls
    # back to computing every pairwise correlation individually in a Python-
    # level loop. Measured directly: method="pearson" on 2,000 columns took
    # 154 seconds; method="spearman" (an earlier version of this function
    # used it, reasoning that ranking predictor relevance doesn't need
    # Pearson's linearity assumption) is far slower still — at only 150
    # columns it already took 9+ seconds, extrapolating to roughly 10
    # minutes at 1,200 columns and far longer beyond that. This is almost
    # certainly what a system crash/hang after "10+ minutes, never reached
    # the actual MICE iterations" was — this function runs BEFORE the
    # ImputationKernel is even constructed, so a hang here presents exactly
    # like that. Column count here can plausibly run into the thousands if
    # COVERAGE_THRESHOLD/the degenerate filters are loosened to keep more
    # columns (the entire point of this function), so this is not a
    # hypothetical edge case — it is the expected way this gets used.
    #
    # Fix: mean-fill (once, cheaply) then a single np.corrcoef call — a
    # proper vectorized matrix operation regardless of NaN, regardless of
    # column count. Confirmed against the same 2,000-column timing case:
    # 1.3 seconds, ~100x faster than pandas' pearson path alone. Losing
    # Spearman's rank-based robustness here is an acceptable trade — this
    # correlation is only ever used to roughly rank candidate predictors
    # for a speed cap, never shown to the user or used for any statistical
    # claim.
    filled = numeric_proxy.to_numpy(dtype=float, copy=True)
    col_means = np.nanmean(filled, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)  # an all-NaN column: fill with 0, corr will be 0 anyway
    nan_mask = np.isnan(filled)
    filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(filled, rowvar=False)
    corr = np.abs(np.nan_to_num(corr, nan=0.0))  # a constant column -> 0/0 -> NaN; treat as uninformative
    corr_df = pd.DataFrame(corr, index=df.columns, columns=df.columns)

    schema = {}
    for target in targets:
        others = [c for c in df.columns if c != target]
        ranked = corr_df.loc[target, others].sort_values(ascending=False)
        schema[target] = ranked.index[:max_predictors].tolist()
    return schema

def run_mice(df: pd.DataFrame, rank_maps: dict[str, dict], columns: dict[str, list[str]]):
    """Fits `N_DATASETS` chains of MICE for `N_ITERATIONS` iterations each
    via miceforest, then snaps imputed ordinal values back to the nearest
    valid rank (a LightGBM regressor imputing a rank column can output a
    non-integer or slightly out-of-range value; ordinal columns were kept
    numeric rather than categorical specifically so this rounding step is
    possible — see the module docstring). Returns the list of M completed
    DataFrames, matching the "M iterations" you asked about for downstream
    per-imputation model fitting / prediction ensembling.
    """
    import miceforest as mf  # deferred import: only required for this step

    if MICE_VERBOSE and VERBOSE_PRINT_EVERY > 1:
        _throttle_miceforest_verbose(VERBOSE_PRINT_EVERY)

    # mean_match_strategy="fast" for binary/nominal columns specifically —
    # NOT a workaround, a structural fix. miceforest's default ("normal")
    # mean-matching for any non-numeric target runs candidate/bachelor
    # predicted probabilities through miceforest/utils.py's logodds():
    #   odds_ratio = probability / (1 - probability); log(odds_ratio)
    # A predicted probability of exactly 0.0 or 1.0 (routine for a lopsided
    # binary column — confirmed against real data on BBRR1_FED_UG_DFLT,
    # one of several BBRR default/discharge-rate columns that end up
    # binary via the exactly-2-realized-values rule above once filtered
    # down to a small surviving population) divides by zero. numpy doesn't
    # raise Python's ZeroDivisionError for float division — it warns and
    # returns inf — so the try/except around that division in miceforest's
    # own source never actually fires; the inf instead propagates into a
    # later "data must be finite" ValueError once something (e.g. the
    # nearest-neighbor donor search) tries to use it.
    #
    # "fast" mean matching for a binary/categorical target dispatches to
    # _mean_match_binary_fast/_mean_match_multiclass_fast instead, which
    # select directly from the predicted probability (argmax, or sampled
    # weighted by probability) and never call logodds() at all — this
    # isn't "less likely to hit the bug," the code path that divides by
    # zero is structurally unreachable for these columns once set. The
    # trade-off, stated plainly: "fast" imputes a binary/nominal value from
    # the model's own prediction directly rather than borrowing an actual
    # observed donor row's value the way nearest-neighbor donor matching
    # does for "normal"/continuous columns — a real behavioral difference,
    # not a free lunch, but a standard, well-supported miceforest option
    # for exactly this variable type, not a monkeypatch or a hack.
    # Ordinal/continuous columns are untouched (left on the "normal"
    # default) — modeled_numeric_columns never runs through logodds() in
    # the first place (see _impute_with_predictions), so they were never
    # at risk from this specific failure mode.
    mean_match_strategy = {col: "fast" for col in columns["binary"] + columns["nominal"]}

    # See build_variable_schema()'s docstring — None (the default) preserves
    # miceforest's own default of every-other-column-as-predictor; set
    # MAX_PREDICTORS_PER_VARIABLE to cap per-model training cost as you add
    # columns back via COVERAGE_THRESHOLD/the degenerate-column filters.
    variable_schema = build_variable_schema(df, MAX_PREDICTORS_PER_VARIABLE)

    # miceforest does not reliably preserve a custom pandas Index (UNITID,
    # set by load_raw()) across versions — confirmed one installed version
    # of miceforest hard-asserts `isinstance(working_data.index, RangeIndex)`
    # at kernel construction and would raise immediately on a UNITID-indexed
    # df; a different installed version ran successfully but produced
    # completed data with no UNITID at all, meaning it silently discarded a
    # non-RangeIndex rather than erroring. Either way, the index is gone
    # before MICE starts, not lost afterward — nothing downstream can
    # "recombine" it back in. Fix: carry UNITID alongside as a plain array
    # instead of relying on it surviving as the DataFrame's actual pandas
    # Index. This is safe because complete_data() only ever copies
    # self.working_data and fills missing values via
    # `.loc[na_where, variable]` (confirmed against miceforest's source) —
    # it never reorders rows, so reattaching by position after the fact is
    # exact, not approximate.
    unitids = df.index.to_numpy()
    df = df.reset_index(drop=True)

    kernel = mf.ImputationKernel(
        df,
        num_datasets=N_DATASETS,
        random_state=RANDOM_STATE,
        mean_match_strategy=mean_match_strategy,
        variable_schema=variable_schema,
    )
    # verbose=MICE_VERBOSE: when True, prints which dataset/iteration/
    # variable it's on — if anything ever crashes mid-run again, this is
    # what lets you see where, since the traceback alone doesn't say when
    # the failure happens inside LightGBM's C extension. Costs negligible
    # Python-side time; see MICE_VERBOSE's docstring on the OTHER cost this
    # has nothing to do with speed — a browser-rendered notebook's output
    # accumulating thousands of printed lines across a full run.
    #
    # num_threads=LGB_NUM_THREADS (-1 by default = all cores): see that
    # constant's docstring — this was hardcoded to 1 earlier in this
    # project to rule out a threading race as the cause of the access-
    # violation crash. That crash's real causes (the "PS" sentinel and
    # logodds() bugs, both now fixed) were unrelated to threading, so
    # LightGBM's normal multi-core tree building was very likely the single
    # largest unnecessary slowdown in every prior run of this pipeline.
    kernel.mice(N_ITERATIONS, verbose=MICE_VERBOSE, num_threads=LGB_NUM_THREADS)

    completed = []
    for i in range(N_DATASETS):
        d = kernel.complete_data(dataset=i)
        for var, rank_map in rank_maps.items():
            if var not in d.columns:
                continue
            lo, hi = min(rank_map.values()), max(rank_map.values())
            d[var] = d[var].round().clip(lower=lo, upper=hi)
        d.index = unitids
        d.index.name = "UNITID"
        completed.append(d)
    return kernel, completed


def run_mice_one_dataset(
    df: pd.DataFrame,
    rank_maps: dict[str, dict],
    columns: dict[str, list[str]],
    dataset_index: int,
    checkpoint_path: Path,
):
    """Runs ONE of the M=N_DATASETS imputation chains in isolation
    (num_datasets=1) instead of all of them together in one process — the
    actual fix for an out-of-memory run, not just a smaller/safer-feeling
    one. See DATASET_INDEX's docstring for the config-level summary; this
    docstring covers the mechanism and the checkpointing behavior.

    Why isolating one dataset reduces peak memory: miceforest's
    ImputationKernel holds the full working-data copy for EVERY dataset in
    num_datasets simultaneously for the whole run, even though its .mice()
    loop only actively trains one dataset's models at a time (confirmed
    against its source: `for dataset in self.datasets: self.complete_data
    (dataset=dataset, inplace=True); for variable: ...` — the other
    datasets' working copies sit in memory the entire time, never released
    between turns). num_datasets=5 therefore costs roughly 5x a single
    dataset's peak memory IN ONE PROCESS. Calling this function 5 times,
    each as a genuine process restart (not 5 calls in one long-running
    session — nothing is freed between calls in the same process), caps
    peak memory at roughly 1x a single dataset's footprint regardless of
    how many total chains (M) you want, because a process restart is what
    actually returns memory to the OS.

    Checkpointing (separate benefit from the above, not a substitute for
    it): pickles the kernel to checkpoint_path after EVERY individual
    iteration via kernel.mice(1, ...) in a loop, rather than one
    kernel.mice(N_ITERATIONS, ...) call. If interrupted — killed, crashed,
    notebook closed — re-running this function with the same
    checkpoint_path reloads the saved kernel and continues from
    kernel.iteration_count() rather than re-doing the whole chain. This
    protects progress WITHIN one dataset's run; it does not by itself
    reduce peak memory the way running one dataset per process does — a
    resumed kernel still holds that one dataset's full working state in
    memory the same as an uninterrupted run would.

    miceforest has no save_kernel()/load_kernel() method (an earlier
    version of this pipeline called one that doesn't exist, in main() —
    now fixed). It implements __getstate__/__setstate__ specifically so
    the kernel pickles with the standard library's pickle module, which is
    what's used here and in main().
    """
    import miceforest as mf  # deferred import: only required for this step

    if MICE_VERBOSE and VERBOSE_PRINT_EVERY > 1:
        _throttle_miceforest_verbose(VERBOSE_PRINT_EVERY)

    # See run_mice()'s matching comment for the full explanation — miceforest
    # does not reliably preserve a custom pandas Index (UNITID) across
    # versions, so it's carried alongside as a plain array and reattached by
    # position afterward instead. Captured unconditionally, before the
    # resume/fresh-start branch below, since build_analysis_frame() is
    # deterministic — main_chunked() hands this function the same row order
    # every time regardless of whether this call is resuming a checkpoint
    # or starting fresh, so unitids stays valid either way.
    unitids = df.index.to_numpy()

    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            kernel = pickle.load(f)
        print(f"Dataset {dataset_index}: resumed from checkpoint at iteration {kernel.iteration_count()}.")
    else:
        # Same mean_match_strategy/variable_schema reasoning as run_mice()
        # above — see that function's docstrings for why each exists.
        mean_match_strategy = {col: "fast" for col in columns["binary"] + columns["nominal"]}
        variable_schema = build_variable_schema(df, MAX_PREDICTORS_PER_VARIABLE)
        kernel = mf.ImputationKernel(
            df.reset_index(drop=True),
            num_datasets=1,
            # RANDOM_STATE + dataset_index, not RANDOM_STATE alone — these
            # five single-dataset kernels need distinct seeds or they'd all
            # produce the identical chain, defeating the purpose of M=5.
            random_state=RANDOM_STATE + dataset_index,
            mean_match_strategy=mean_match_strategy,
            variable_schema=variable_schema,
        )

    while kernel.iteration_count() < N_ITERATIONS:
        kernel.mice(1, verbose=MICE_VERBOSE, num_threads=LGB_NUM_THREADS)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(kernel, f)
        print(f"Dataset {dataset_index}: checkpointed after iteration {kernel.iteration_count()}/{N_ITERATIONS}.")

    completed = kernel.complete_data(dataset=0)
    for var, rank_map in rank_maps.items():
        if var not in completed.columns:
            continue
        lo, hi = min(rank_map.values()), max(rank_map.values())
        completed[var] = completed[var].round().clip(lower=lo, upper=hi)
    completed.index = unitids
    completed.index.name = "UNITID"
    return kernel, completed


def main_chunked(dataset_index: int):
    """Entry point for running ONE of the M=N_DATASETS chains in isolation
    — call this once per dataset_index (0 through N_DATASETS-1), each as a
    genuine process restart for the memory benefit in run_mice_one_dataset()
    to actually materialize. Running all N_DATASETS calls back-to-back
    inside one long-lived process defeats the purpose, since nothing gets
    released between them.

    From a terminal (recommended — see MICE_VERBOSE's docstring on why a
    notebook's rendered output, not necessarily the Python process itself,
    is a likely explanation for a browser-reported "out of memory"):
        python mice_pipeline.py 0
        python mice_pipeline.py 1
        ...
        python mice_pipeline.py 4
    Or from Jupyter/VSCode, restarting the kernel between each call:
        main_chunked(0)   # then restart the kernel
        main_chunked(1)   # then restart the kernel
        ...
    Each call re-runs build_analysis_frame() from scratch — cheap (seconds)
    relative to the MICE run itself, and necessary since nothing persists
    across the process restart that makes the memory benefit real.
    After all N_DATASETS calls, run check_chunked_datasets() to confirm
    every piece landed before handing off to post_mice_modeling.py.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    df, report, rank_maps = build_analysis_frame(RAW_CSV)
    checkpoint_path = OUTPUT_DIR / f"checkpoint_dataset_{dataset_index}.pkl"
    kernel, completed = run_mice_one_dataset(
        df, rank_maps, report["columns"], dataset_index, checkpoint_path
    )
    completed.to_csv(OUTPUT_DIR / f"completed_{dataset_index}.csv", index=True)
    print(f"Dataset {dataset_index} complete — wrote completed_{dataset_index}.csv")


def check_chunked_datasets() -> bool:
    """Run after all N_DATASETS main_chunked() calls. There is no real
    "recombining" step needed — each completed_{i}.csv is already a
    complete, standalone table (miceforest's M chains were always
    independent of each other, chunked or not) — this just confirms every
    piece landed in the layout post_mice_modeling.py already expects
    (completed_0.csv ... completed_{M-1}.csv), so a half-finished chunked
    run doesn't silently look done."""
    missing = [i for i in range(N_DATASETS) if not (OUTPUT_DIR / f"completed_{i}.csv").exists()]
    if missing:
        print(f"Still missing dataset(s): {missing} — run main_chunked(i) for each, or `python mice_pipeline.py {missing[0]}`.")
        return False
    print(f"All {N_DATASETS} completed datasets present in {OUTPUT_DIR}/ — ready for post_mice_modeling.py.")
    return True


def find_crashed_variable(exc: BaseException) -> list[str]:
    """Identifies which variable's LightGBM training was in progress when
    kernel.mice() crashed, by walking the exception's traceback to find
    ImputationKernel.mice()'s local `logger` variable and reading its
    `started_timers` dict.

    This isn't a guess: confirmed against miceforest's actual source
    (imputation_kernel.py / logger.py). Inside .mice(), immediately before
    each variable's LightGBM training call, it runs:
        time_key = dataset, iteration, variable, "Training"
        logger.set_start_time(time_key)
        current_model = train(...)          # <- the crash happens here
        logger.record_time(time_key)        # <- never reached if it crashes
    `set_start_time` adds time_key to `logger.started_timers`;
    `record_time` removes it. So at the moment of a crash, exactly one key
    is left in `started_timers` — the (dataset, iteration, variable, event)
    tuple for whatever was mid-training. Since miceforest only appends its
    Logger to kernel.loggers AFTER .mice() returns successfully (see its
    source), that logger is otherwise unreachable once the exception has
    propagated — the traceback's stack frames are the only remaining
    reference to it, which is why this walks tb_frame.f_locals instead of
    going through the kernel object.

    Returns the variable name(s) still marked "started" (normally exactly
    one, given num_threads=1 / sequential processing) — empty list if the
    logger couldn't be found (e.g. miceforest changes this internal
    structure in a future version).
    """
    tb = exc.__traceback__
    while tb is not None:
        local_logger = tb.tb_frame.f_locals.get("logger")
        if local_logger is not None and hasattr(local_logger, "started_timers"):
            if local_logger.started_timers:
                return [key[2] for key in local_logger.started_timers]
        tb = tb.tb_next
    return []


def run_mice_with_retry(
    df: pd.DataFrame,
    rank_maps: dict[str, dict],
    columns: dict[str, list[str]],
    max_attempts: int = 10,
):
    """Stopgap wrapper around run_mice() for while a run still crashes
    intermittently on some not-yet-identified column: catches an OSError
    (the access-violation pattern this project has been chasing), uses
    find_crashed_variable() to identify exactly which column was mid-
    training, drops it, and retries — up to max_attempts times.

    This is NOT a substitute for drop_degenerate_columns/
    drop_near_constant_continuous, which are the preventative fix — a
    column removed here should be treated as a signal that those
    thresholds (MIN_CATEGORY_COUNT/FRACTION, MAX_DOMINANT_VALUE_SHARE)
    might need tightening, or that the column deserves a specific look,
    not just silently accepted. Every column this loop had to remove is
    returned so that decision stays visible rather than disappearing into
    a successful-looking run.
    """
    working_df = df.copy()
    working_columns = {k: list(v) for k, v in columns.items()}
    removed: list[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            kernel, completed = run_mice(working_df, rank_maps, working_columns)
            if removed:
                print(f"Succeeded after removing {len(removed)} column(s): {removed}")
            return kernel, completed, removed
        except OSError as exc:
            crashed_vars = find_crashed_variable(exc)
            if not crashed_vars:
                print("Crash occurred but the crashed variable could not be identified "
                      "from the logger (see find_crashed_variable's docstring) — "
                      "stopping automatic retry rather than guessing.")
                raise
            print(f"Attempt {attempt}/{max_attempts}: crashed while training "
                  f"{crashed_vars} — removing and retrying.")
            for var in crashed_vars:
                if var not in working_df.columns:
                    continue
                working_df = working_df.drop(columns=[var])
                removed.append(var)
                for group in working_columns.values():
                    if var in group:
                        group.remove(var)

    raise RuntimeError(
        f"Still crashing after {max_attempts} attempts. Removed so far: {removed}. "
        "Either raise max_attempts, or this may not be a single-column problem — "
        "worth checking pip-installed lightgbm/miceforest versions at this point."
    )


# ────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────

def ensure_reference_table(
    raw_csv: str | Path, dictionary_csv: str | Path, path: Path | None = None
) -> None:
    """Builds the UNITID -> INSTNM lookup at `path` if one doesn't already
    exist there; does nothing (not even a re-read of raw_csv) if it does.
    Safe to call at the top of every run, including every one of the
    N_DATASETS separate processes in Option B — it only ever does real work
    once.

    path defaults to None (resolved to the module-level REFERENCE_TABLE_PATH
    below) rather than `= REFERENCE_TABLE_PATH` directly in the signature —
    same reasoning as load_raw()'s dictionary_csv parameter, and caught by
    the same mistake during this function's own testing: a plain default
    like that is evaluated once, when the function is defined, so
    reassigning REFERENCE_TABLE_PATH afterward would silently have no
    effect on calls that rely on the default.

    Peeks at just the header row to resolve column names (handles both the
    flat convention and the dotted one the same way load_raw() does, via
    build_dotted_to_flat()) rather than loading the full multi-thousand-
    column file a second time just to pull two columns out of it.
    """
    if path is None:
        path = REFERENCE_TABLE_PATH
    if path.exists():
        print(f"Reference table already exists at {path} — leaving it as-is.")
        return

    header = pd.read_csv(raw_csv, nrows=0).columns.tolist()
    dotted_to_flat = build_dotted_to_flat(dictionary_csv)
    flat_header = [dotted_to_flat.get(c, c) for c in header]

    wanted = {"UNITID", "INSTNM"}
    usecols = [orig for orig, flat in zip(header, flat_header) if flat in wanted]
    if len(usecols) < 2:
        print(f"WARNING: could not find both UNITID and INSTNM in {raw_csv} "
              f"(found: {usecols}) — reference table not created.")
        return

    ref = pd.read_csv(raw_csv, usecols=usecols, na_values=["PrivacySuppressed", "NULL", "PS"])
    ref = ref.rename(columns=dict(zip(usecols, [dotted_to_flat.get(c, c) for c in usecols])))
    ref = ref[["UNITID", "INSTNM"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(path, index=False)
    print(f"Created reference table ({len(ref)} institutions) at {path}.")


def build_analysis_frame(
    raw_csv: str | Path | None = None,
    dictionary_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Everything up through column classification/typing — no miceforest
    dependency required. Useful on its own to inspect the coverage report
    and column classification before committing to a MICE run.

    Both paths default to None (resolved to the module-level RAW_CSV /
    DICTIONARY_CSV below), for the same reason as load_raw() above — see
    its docstring.
    """
    if raw_csv is None:
        raw_csv = RAW_CSV
    if dictionary_csv is None:
        dictionary_csv = DICTIONARY_CSV

    ensure_reference_table(raw_csv, dictionary_csv)

    df = load_raw(raw_csv, dictionary_csv)

    df, metadata_df, metadata_cols = split_metadata_columns(df)
    if metadata_cols:
        print(f"Set aside {len(metadata_cols)} per-variable metadata column(s) "
              f"(matching {METADATA_COLUMN_SUFFIXES}) — excluded from MICE, not deleted; "
              f"see report['metadata_df'].")

    df, rank_maps = apply_structural_fixes(df)

    # Checkpoint: sparsity of the actual MICE candidate set, now that
    # metadata columns are out and structural fixes have corrected the
    # missingness (e.g. Carnegie '-2' no longer counted as "present") —
    # computed BEFORE select_by_coverage decides pass/fail, so the full
    # distribution is inspectable rather than only the after-the-fact counts.
    sparsity_report = report_sparsity(df)

    if APPLY_COVERAGE_FILTER:
        df, dropped = select_by_coverage(df, COVERAGE_THRESHOLD)
    else:
        dropped = pd.Series(dtype=float)

    # Classify BEFORE the degenerate-category check, not after: that check
    # only makes sense for columns headed for classification (nominal/
    # binary) — an ordinal column with a rare rank isn't a crash risk the
    # same way a classification target with a near-empty class is (see
    # drop_degenerate_columns' docstring), so it needs the classification
    # split to know which columns to even look at.
    columns = classify_columns(df)
    df, dropped_degenerate = drop_degenerate_columns(df, columns["nominal"] + columns["binary"])
    dropped_degenerate_set = set(dropped_degenerate)
    columns["nominal"] = [c for c in columns["nominal"] if c not in dropped_degenerate_set]
    columns["binary"] = [c for c in columns["binary"] if c not in dropped_degenerate_set]

    df, dropped_near_constant = drop_near_constant_continuous(df, columns["continuous"])
    dropped_near_constant_set = set(dropped_near_constant)
    columns["continuous"] = [c for c in columns["continuous"] if c not in dropped_near_constant_set]

    df = finalize_dtypes(df, columns)
    return df, {
        "dropped_for_coverage": dropped,
        "dropped_for_low_variance": dropped_degenerate + dropped_near_constant,
        "columns": columns,
        "metadata_columns": metadata_cols,
        "metadata_df": metadata_df,
        "sparsity_report": sparsity_report,
    }, rank_maps


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df, report, rank_maps = build_analysis_frame(RAW_CSV)

    if report["metadata_columns"]:
        print(f"Metadata columns set aside (not modeled, not deleted): {len(report['metadata_columns'])}")
        report["metadata_df"].to_csv(OUTPUT_DIR / "metadata_columns.csv", index=True)

    print("Sparsity checkpoint (before coverage filtering) — worst 15 of the MICE candidate columns:")
    print(report["sparsity_report"].head(15).to_string(index=False))
    report["sparsity_report"].to_csv(OUTPUT_DIR / "sparsity_report.csv", index=False)

    print(f"Columns kept: {df.shape[1]}  |  rows: {df.shape[0]}")
    print(f"Columns dropped for coverage < {COVERAGE_THRESHOLD:.0%}: {len(report['dropped_for_coverage'])}")
    print(f"Columns dropped for a category with < {MIN_CATEGORY_COUNT} rows: {len(report['dropped_for_low_variance'])}")
    for kind, cols in report["columns"].items():
        print(f"  {kind}: {len(cols)}")

    report["dropped_for_coverage"].to_csv(OUTPUT_DIR / "dropped_for_coverage.csv", header=["coverage"])
    (OUTPUT_DIR / "dropped_for_low_variance.json").write_text(json.dumps(report["dropped_for_low_variance"], indent=2))
    (OUTPUT_DIR / "column_classification.json").write_text(json.dumps(report["columns"], indent=2))

    kernel, completed, removed_by_retry = run_mice_with_retry(df, rank_maps, report["columns"])
    if removed_by_retry:
        (OUTPUT_DIR / "dropped_by_crash_retry.json").write_text(json.dumps(removed_by_retry, indent=2))
        print(f"NOTE: {len(removed_by_retry)} column(s) were removed reactively after crashing "
              f"mid-run rather than being caught by the preventative filters — see "
              f"dropped_by_crash_retry.json, and consider whether MIN_CATEGORY_COUNT/"
              f"MIN_CATEGORY_FRACTION/MAX_DOMINANT_VALUE_SHARE should be tightened.")

    for i, d in enumerate(completed):
        d.to_csv(OUTPUT_DIR / f"completed_{i}.csv", index=True)  # index=UNITID — needed to join a DV on afterward
    # miceforest has no save_kernel()/load_kernel() method — it implements
    # __getstate__/__setstate__ specifically so the kernel pickles with the
    # standard library's pickle module (confirmed against its source). An
    # earlier version of this line called a method that doesn't exist and
    # would have raised AttributeError here, after every other output had
    # already written successfully.
    with open(OUTPUT_DIR / "mice_kernel.pkl", "wb") as f:
        pickle.dump(kernel, f)

    print(f"Wrote {N_DATASETS} completed datasets to {OUTPUT_DIR}/")


if __name__ == "__main__":
    # `python mice_pipeline.py 2` runs ONLY dataset chain 2 in isolation
    # (see main_chunked()'s docstring) — a CLI argument always takes
    # precedence over DATASET_INDEX so you can chunk without editing the
    # file between each of the N_DATASETS runs. With no argument: falls
    # back to DATASET_INDEX if set, otherwise runs the original
    # all-datasets-in-one-process main().
    if len(sys.argv) > 1:
        main_chunked(int(sys.argv[1]))
    elif DATASET_INDEX is not None:
        main_chunked(DATASET_INDEX)
    else:
        sys.exit(main())
