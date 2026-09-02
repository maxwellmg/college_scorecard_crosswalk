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

COVERAGE_THRESHOLD = 0.85   # the "magic proportion" — min non-missing share to keep a column
N_DATASETS = 5              # M — number of completed datasets MICE produces
N_ITERATIONS = 5            # MICE iterations per dataset
RANDOM_STATE = 42

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
MIN_CATEGORY_COUNT = 20

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
    to real NaN at read time. `NA` is caught by pandas' default na_values;
    `PrivacySuppressed`/`NULL` are added defensively for other Scorecard
    files even though neither occurs in the flat-named extract this was
    first verified against.

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
        na_values=["PrivacySuppressed", "NULL"],
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
    'Other'."""
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

def drop_degenerate_columns(df: pd.DataFrame, min_category_count: int = MIN_CATEGORY_COUNT) -> tuple[pd.DataFrame, list[str]]:
    """Drop any non-numeric (nominal/binary-typed-as-object) column whose
    2nd-most-common realized value has fewer than min_category_count rows —
    the condition that produces miceforest's "very rare categories" warning
    and, in the worst case, a LightGBM training crash on a near-empty class.
    Columns in PROTECTED_FROM_VARIANCE_DROP are exempt (see its docstring).

    Runs on raw dtypes (object/str/numeric-with-few-uniques), i.e. before
    finalize_dtypes casts anything to 'category' — this only needs to
    inspect realized value counts, not final dtype.
    """
    dropped = []
    for col in df.columns:
        if col in PROTECTED_FROM_VARIANCE_DROP:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 20:
            continue  # treat as continuous-ish; not the target of this check
        counts = df[col].value_counts(dropna=True)
        if counts.shape[0] < 2:
            dropped.append(col)  # constant or entirely missing — useless either way
            continue
        if counts.iloc[1] < min_category_count:
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

    continuous_cols = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]
    nominal_cols += [c for c in remaining if c not in continuous_cols]

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

    kernel = mf.ImputationKernel(
        df,
        num_datasets=N_DATASETS,
        random_state=RANDOM_STATE,
    )
    # verbose=True: if anything ever crashes mid-run again, this prints which
    # dataset/iteration/variable it was on right before the crash — the
    # traceback alone doesn't say, since the failure happens inside
    # LightGBM's C extension, past the point Python could report it.
    # num_threads=1: LightGBM + OpenMP crashing with a null-pointer access
    # violation under multithreading is a known failure mode on Windows/
    # Anaconda in particular. If drop_degenerate_columns() doesn't fully
    # resolve the crash, this is the next thing to try — slower, but a
    # useful bisection step to confirm whether it's a threading race rather
    # than a data problem. Safe to remove once you've confirmed it's stable.
    kernel.mice(N_ITERATIONS, verbose=True, num_threads=1)

    completed = []
    for i in range(N_DATASETS):
        d = kernel.complete_data(dataset=i)
        for var, rank_map in rank_maps.items():
            if var not in d.columns:
                continue
            lo, hi = min(rank_map.values()), max(rank_map.values())
            d[var] = d[var].round().clip(lower=lo, upper=hi)
        completed.append(d)
    return kernel, completed


# ────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────

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
    df = load_raw(raw_csv, dictionary_csv)
    df, rank_maps = apply_structural_fixes(df)

    if APPLY_COVERAGE_FILTER:
        df, dropped = select_by_coverage(df, COVERAGE_THRESHOLD)
    else:
        dropped = pd.Series(dtype=float)

    df, dropped_degenerate = drop_degenerate_columns(df)

    columns = classify_columns(df)
    df = finalize_dtypes(df, columns)
    return df, {
        "dropped_for_coverage": dropped,
        "dropped_for_low_variance": dropped_degenerate,
        "columns": columns,
    }, rank_maps


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df, report, rank_maps = build_analysis_frame(RAW_CSV)

    print(f"Columns kept: {df.shape[1]}  |  rows: {df.shape[0]}")
    print(f"Columns dropped for coverage < {COVERAGE_THRESHOLD:.0%}: {len(report['dropped_for_coverage'])}")
    print(f"Columns dropped for a category with < {MIN_CATEGORY_COUNT} rows: {len(report['dropped_for_low_variance'])}")
    for kind, cols in report["columns"].items():
        print(f"  {kind}: {len(cols)}")

    report["dropped_for_coverage"].to_csv(OUTPUT_DIR / "dropped_for_coverage.csv", header=["coverage"])
    (OUTPUT_DIR / "dropped_for_low_variance.json").write_text(json.dumps(report["dropped_for_low_variance"], indent=2))
    (OUTPUT_DIR / "column_classification.json").write_text(json.dumps(report["columns"], indent=2))

    kernel, completed = run_mice(df, rank_maps, report["columns"])
    for i, d in enumerate(completed):
        d.to_csv(OUTPUT_DIR / f"completed_{i}.csv", index=True)  # index=UNITID — needed to join a DV on afterward
    kernel.save_kernel(str(OUTPUT_DIR / "mice_kernel.pkl"))

    print(f"Wrote {N_DATASETS} completed datasets to {OUTPUT_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
