# Hand-off: College Scorecard MICE — crash debugging in progress

**Read this first if you're picking up this project in a new session.** This
is not a general project README (see `README_MICE.md` and
`README_POST_MICE_MODELING.md` for that) — it's a status report on a
specific, still-unresolved bug that's blocked a working MICE run for
multiple sessions. Its job is to stop you from re-suggesting things that
have already been tried and ruled out, and to tell you exactly what
evidence is still needed.

---

## 1. Where things stand, in one paragraph

MICE (via `miceforest`, LightGBM-backed) will not complete a run on the
user's real ~10-year historical College Scorecard pull. It fails with
`OSError: exception: access violation reading 0x0000000000000000` — a
native crash inside LightGBM's C extension, on Windows, deep in
`Dataset._lazy_init` → `set_label` → `set_field` →
`_LIB.LGBM_DatasetSetField`. A retry mechanism (`run_mice_with_retry` in
`mice_pipeline.py`) correctly identifies and removes whichever column was
mid-training when it crashes, then retries — but it never converges. The
most recent 10-attempt run revealed the key fact that reframes the whole
investigation (§3.7 below): **every single attempt crashed on the very
first variable trained, regardless of which variable that was**, across 10
columns with no common data profile. That points away from "some columns
have bad data" and toward an environment/process-level cause.

**The user was told to run a completely isolated LightGBM smoke test
(`lightgbm_smoke_test.py`, in this repo) in a FRESH kernel restart, before
touching any real data, to determine whether LightGBM itself is broken in
this environment.** That result had not come back as of this hand-off. If
you're picking this up, **that's the first thing to ask for.**

---

## 2. The immediate next step — get this result first

Ask the user: *did you run `lightgbm_smoke_test.py` in a freshly restarted
Jupyter kernel, and what happened?*

- **If it crashed too** (same `OSError: access violation`, on ~200 rows of
  made-up random numbers with zero relationship to Scorecard data): this
  conclusively proves the problem is the LightGBM install/environment
  itself, not this project's code or data. Next steps, in order of how
  cheap/likely-to-work they are:
  1. `pip uninstall lightgbm` then `conda install -c conda-forge lightgbm`
     instead of the pip wheel — conda-forge's Windows build bundles its own
     OpenMP runtime and avoids DLL conflicts pip wheels sometimes hit. This
     is the single most likely fix.
  2. Check for a duplicate/conflicting `lib_lightgbm.dll` somewhere else on
     the system `PATH` (a real, documented cause of this exact error — see
     the sources cited in §4).
  3. Check whether antivirus / endpoint protection (this looks like a
     corporate-managed Windows machine — `AppData\Local\anaconda3`, a
     `MMiller1` user profile) is intercepting the native DLL load.
  4. If none of that resolves it, the pragmatic fallback is R's `mice`
     package (see §5 — prep work for that is already partly done).

- **If it succeeded cleanly**: the LightGBM install itself is fine, which
  means the "always position 1" pattern is better explained by **a first
  native crash corrupting that Python process's memory, such that every
  subsequent LightGBM call in the *same process* is unreliable regardless
  of input** — not a fresh-kernel-object problem, a fresh-*process* problem.
  `run_mice_with_retry`'s current design (catch → drop column → retry in
  the same process) doesn't account for this. The fix is to make each
  retry attempt run in a genuinely separate OS process (e.g. `subprocess`
  or `multiprocessing` with a fresh interpreter per attempt), so a crash in
  one attempt can't contaminate the next. **This has not been built yet** —
  it's the next thing to write if the smoke test comes back clean.

---

## 3. What's already been tried and ruled out — do not re-suggest these

In roughly chronological order:

1. **Naming convention (dotted vs. flat).** The real data file uses the
   College Scorecard API's dotted "developer-friendly name" convention
   (`school.ownership`, `admissions.test_requirements`, `academics.program...`)
   rather than the flat `VARIABLE NAME` convention (`CONTROL`, `PREDDEG`).
   **Solved** — `mice_pipeline.py`'s `build_dotted_to_flat()` /
   `rename_to_flat_convention()` derive the crosswalk programmatically from
   `CollegeScorecardDataDictionary.csv` and rename on load. Confirmed
   working in production (one run renamed 1,818 columns correctly).

2. **"Maybe there's merged-in external data with bad columns."** Raised
   after misreading blurry crash-log screenshots as containing names like
   `GRANT_NAME_10` / `LIFTUPFOOD_ARR` / `DARK_THR_STATE` — those names were
   almost certainly a misread of a low-quality phone photo (inconsistent
   even within that response), and **the user has explicitly confirmed the
   MICE input contains only genuine College Scorecard variables.** This
   theory is dead. Don't revive it without new, clearly-legible evidence.

3. **OpenMP threading race.** `num_threads=1` was added to the
   `kernel.mice()` call specifically to rule this out. **The crash persisted
   with `num_threads=1` active** — threading race is ruled out as the sole
   cause.

4. **LightGBM version.** The user changed their installed LightGBM version
   (from 4.7.0 — exact new version not recorded before this hand-off,
   **ask again**) and **still got the same crash pattern**. A pure
   version-pin fix has already been tried once and didn't resolve it — that
   doesn't rule out version issues entirely (the smoke test may still
   implicate the install), but don't treat "try a different lightgbm
   version" alone as a fresh idea.

5. **A known miceforest/LightGBM compatibility bug** — miceforest issue #95
   (`lightgbm 4.6.0 removed support for categorical_feature in lgb.train,
   breaking miceforest`) was investigated. It's real, but it was fixed in
   miceforest ≥6.0.3, which the user already had. **Not the cause here**,
   though it established that this exact package pairing has a track record
   of fragile version interactions, which is why the smoke test in §2
   matters.

6. **Column-level statistical instability** (sparse/rare categories, near-
   constant continuous columns). Real, and worth having fixed regardless —
   implemented as `drop_degenerate_columns` (categorical: drops a column if
   its 2nd-most-common realized category fails both an absolute floor
   `MIN_CATEGORY_COUNT` and a relative floor `MIN_CATEGORY_FRACTION`) and
   `drop_near_constant_continuous` (numeric: drops a column if one value
   accounts for ≥`MAX_DOMINANT_VALUE_SHARE` of it). These **reduced** crash
   frequency across sessions but did **not** eliminate it — crashes kept
   recurring on columns that passed these filters cleanly (e.g. `DOLPROVIDER`,
   a clean 0/1 binary split, still crashed). This is what motivated looking
   past "the data has a rare-category problem" toward the environment.

7. **THE KEY FINDING: crash position is always #1, not the variable
   identity.** In the most recent 10-attempt retry run, the crashing
   variable was different every time (`COUNT_NWNE_1YR`, `EARN_THR_STATE`,
   `OVERALL_YR3_N`, `OVERALL_YR2_N`, `OVERALL_YR4_N`, `DOLPROVIDER`,
   `COUNT_NWNE_P6`, `SCH_DEG`, `MD_EARN_WNE_4YR`, `COUNT_WNE_4YR`) — but
   **every one of them was literally the first variable attempted in that
   retry's fresh `ImputationKernel`**, confirmed via the diagnostic
   instrumentation (see §4). No shared data characteristic across those 10
   columns (dtype float64 or category, nunique from 3 to 3,464, all
   `n_missing=0`). This is the evidence behind the current two-hypothesis
   fork in §2, and is why the investigation moved from "fix the data" to
   "isolate the process."

---

## 4. Diagnostic tooling already built (all in this repo)

| File | What it does | Status |
|---|---|---|
| `mice_pipeline.py` | Full preprocessing pipeline: dotted→flat renaming, structural `-2`/sentinel handling, ordinal encoding, metadata-column split (`*_year_added`), sparsity checkpoint, coverage filtering, degenerate-column dropping, `run_mice`/`run_mice_with_retry`. Domain logic is documented in `README_MICE.md`. | Preprocessing confirmed working correctly; the MICE call itself is what's crashing. |
| `diagnose_mice_crash.py` | Monkeypatches `ImputationKernel._make_features_label` (confirmed exact hook point from miceforest 6.0.3 source) to print per-variable diagnostics (dtype, missing count, category codes/value_counts) and pickle the last-attempted variable's features/label to `mice_diagnostics/last_attempted_variable.pkl` before every training attempt. Idempotent against repeated `import`/`importlib.reload` in a persistent Jupyter kernel (checks a `_is_diagnostic_patch` marker before re-wrapping). | Working — this is what produced the finding in §3.7. `import diagnose_mice_crash` before calling `.mice()`/`main()`. |
| `find_crashed_variable()` (inside `mice_pipeline.py`) | Walks the caught `OSError`'s traceback frames to find `ImputationKernel.mice()`'s local `logger` variable, and reads `logger.started_timers` (a dict miceforest itself maintains — a `(dataset, iteration, variable, "Training")` key is added when training starts and deleted when it finishes) to identify exactly which variable was mid-training when the crash hit. | Working correctly — confirmed identifying the right variable every time across 10 attempts. |
| `lightgbm_smoke_test.py` | Minimal, data-independent LightGBM training test (200 rows of `numpy.random` data, no miceforest, no Scorecard data at all). The decisive test described in §2. | **Written, not yet run** (or run but result not yet reported back as of this hand-off — confirm which). |
| `run_mice_with_retry()` (inside `mice_pipeline.py`) | Catches the `OSError`, uses `find_crashed_variable()`, drops the column, retries, up to `max_attempts`. | Works as designed, but per §3.7 its premise (retry in the same process) may itself be unsound if the crash corrupts process state — do not extend this function further until the §2 fork is resolved. |
| `post_mice_modeling.py` + `README_POST_MICE_MODELING.md` | Downstream step: fits several model types (linear/logistic, Lasso/LassoCV, SVR/SVC, RandomForest) across the `M` completed datasets MICE would produce, ensembles predictions, reports per-imputation score spread. | Built and documented, but **blocked** — can't run until MICE itself produces `completed_*.csv` files. Not the current bottleneck; don't spend time here until §2 resolves. |
| `README_MICE.md` | Full domain/preprocessing reference — ordinal variable list, sentinel-code handling (`-1`/`-2`), RELAFFIL collapsing, cardinality reduction, etc. | Authoritative for anything about *what the data means* or *how it should be encoded*. This hand-off doc is only about the crash. |

**Useful research already done (don't redo):**
- [lightgbm 4.6.0 removed support for categorical_feature in lgb.train, breaking miceforest · Issue #95 · AnotherSamWilson/miceforest](https://github.com/AnotherSamWilson/miceforest/issues/95)
- [OSError: exception: access violation writing 0x0000000000000000 · Issue #4912 · microsoft/LightGBM](https://github.com/microsoft/LightGBM/issues/4912) — cites a duplicate/conflicting `lib_lightgbm.dll` load as one root cause of this exact error class.

---

## 5. If it comes to R (last resort, not yet needed)

The user has said they'd rather not port to R if avoidable — treat this as
the fallback only after §2's fork is resolved and doesn't fix things. Prep
already done for that eventuality (see the "If you still want to move to R"
section of an earlier response, and `README_POST_MICE_MODELING.md`'s design
choices, which transfer directly):
- `mice_pipeline.py` already writes `column_classification.json`
  (ordinal/binary/nominal/continuous per column) — an R import script should
  read this to assign `factor`/`ordered`/numeric types rather than
  reconstructing that classification by hand.
- R's `mice` has a tree-based method (`method = "rf"`, via `ranger`/
  `randomForest`) that's the actual analog to what miceforest does, without
  any LightGBM dependency — if it comes to R, this is the method to reach
  for first, since it sidesteps this entire crash class by construction.

---

## 6. Environment facts to reconfirm (don't assume these are still true)

- `pip show lightgbm miceforest` — versions were `lightgbm==4.7.0`,
  `miceforest==6.0.3` as of the crash that produced the position-1 finding
  in §3.7, but the user has since changed the LightGBM version at least
  once (§3.4) — get the current versions fresh.
- `DICTIONARY_CSV` / `RAW_CSV` in `mice_pipeline.py` are machine-specific
  absolute paths the user edits directly — if this session is on a
  different machine than the last one, these will need to be re-pointed.
  (Both already default via a `None`-sentinel pattern specifically so that
  editing the module-level constant after import takes effect — see the
  comments right on those two functions if this comes up again.)
- `COVERAGE_THRESHOLD`, `MIN_CATEGORY_COUNT`, `MIN_CATEGORY_FRACTION`,
  `MAX_DOMINANT_VALUE_SHARE` have all been tuned across sessions (values
  seen in transcripts: coverage tested at 0.85, 0.50, and 0.90) — check
  what they're currently set to in the shared repo rather than assuming a
  specific value.
