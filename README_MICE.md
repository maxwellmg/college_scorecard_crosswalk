# College Scorecard — Pre-MICE Preprocessing Guide

**Purpose:** hand-off spec for the engineer writing the preprocessing script that
runs *before* MICE (`mice` in R, or `IterativeImputer`/`miceforest` in Python) on
the institution-level College Scorecard data.

**Source of truth used to build this doc:**
`API_Documentation/CollegeScorecardDataDictionary.csv` — the long-format
dictionary with explicit `VALUE`/`LABEL` rows per categorical variable — cross-
checked directly against `Most-Recent-Cohorts-Institution.csv` (the actual
extract; see **Correction** below). The dictionary tells you a variable's
*coding scheme*; `Institution_cohort_map.csv` only tells you which years each
variable exists in and was not used beyond that.

**Correction after reviewing the real data (2026-08-28):** the working file
turned out to be **raw, natively-coded — not one-hot encoded**.
`Most-Recent-Cohorts-Institution.csv` has 3,308 columns, one per dictionary
variable, and each categorical field (`CONTROL`, `PREDDEG`, `LOCALE`, etc.) is
a single column holding its native integer/string code exactly as the
dictionary defines it — not a set of `VAR_1`/`VAR_2`/`VAR_3` dummy columns.
Confirmed by pulling full value counts on every categorical field named
below directly from that file (see per-section notes tagged **[verified]**).

This changes the shape of the job, for the better: there's nothing to *undo*.
The preprocessing script never needs a one-hot-detection step (the original
§0 below) — it just needs to encode each variable the *right* way the first
and only time:

1. **Ordinal variables** (e.g. degree level, urbanicity) should be cast to a
   single ordinal-integer column (with the remaps in §1, several of which are
   *not* a straight cast of the raw code) and handed to MICE as ordinal
   (`polr` in R) or numeric (`pmm`) — never one-hot encoded in the first
   place. One-hot encoding these and letting MICE impute each dummy
   independently is the failure mode this doc exists to head off before it's
   built, not to repair after the fact.
2. **Binary flags** (`HBCU`, `MAIN`, `CURROPER`, etc.) are already single
   0/1 columns in the source — leave them as one column. If a later step
   one-hot encodes them anyway (e.g. a generic "one-hot everything
   categorical" utility function), make sure it drops one level — a 2-column
   dummy pair for a binary flag is perfectly collinear and destabilizes every
   other regression-based imputation model that uses it as a predictor.

§0 below is kept as a defensive check (in case a different extract than the
one reviewed here ever does arrive pre-encoded), but for
`Most-Recent-Cohorts-Institution.csv` specifically, skip straight to §1.

---

## 0. Defensive check only — not needed for `Most-Recent-Cohorts-Institution.csv`

Confirmed by direct inspection: this file is not one-hot encoded, so this
step is a no-op for it. Keeping this section in case a future extract (a
different pull, a downstream copy someone else transformed) *does* arrive
pre-encoded — the script should still guard against that rather than assume
it forever. If a variable is ever found split across multiple `VAR_*`
columns, **detect one-hot groups programmatically** rather than hardcoding
column names:

- Group columns by shared prefix (everything before the last `_`).
- A candidate one-hot group is any prefix where (a) all suffixes are 0/1-valued,
  (b) row sums across the group are 0 or 1 for every row, and (c) the prefix
  matches a `VARIABLE NAME` in the dictionary that has ≥2 `VALUE`/`LABEL` rows.
- Cross-check the recovered category codes against the `VALUE` column for that
  variable in `CollegeScorecardDataDictionary.csv` — this catches silent
  mismatches (e.g. `get_dummies` dropping a category with `drop_first=True`,
  or suffixes being the `LABEL` text instead of the `VALUE` code).

Everything below assumes that detection step recovers, for each flagged
variable, the original coded value per row (or `NaN`/all-zero if the
source was missing before encoding — see §2).

---

## 1. Variables to encode as ORDINAL (never one-hot in the first place)

These have a genuine, defensible rank order. Recommended `mice` method:
`polr` (proportional odds / ordinal logistic) in R, or treat as numeric for
`pmm` if the scale is being used as a covariate rather than an imputation
target.

| Variable | Raw codes → ordinal | Notes / gotchas |
|---|---|---|
| `PREDDEG` | 0 (not classified) < 1 (certificate) < 2 (associate) < 3 (bachelor's) < 4 (graduate-only) | Already monotonic as coded. Direct cast, no remapping needed. **[verified]** all 5 codes present in `Most-Recent-Cohorts-Institution.csv`, no unexpected values. |
| `HIGHDEG` | 0 (non-degree) < 1 (certificate) < 2 (associate) < 3 (bachelor's) < 4 (graduate) | Same as `PREDDEG`. `PREDDEG` and `HIGHDEG` are correlated but not redundant (`HIGHDEG` ≥ `PREDDEG` always) — keep both. |
| `ICLEVEL` | 1 (4-year) < 2 (2-year) < 3 (less-than-2-year) | Coded in decreasing "level," not increasing — that's fine for `polr`, but if you want intuitive direction, reverse-code (`4 - ICLEVEL`) before modeling. Document whichever direction you pick. **[verified]** 3010/1551/1712 rows respectively, no missing/unexpected values. |
| `LOCALE` | Re-rank the 12 raw codes (11,12,13,21,22,23,31,32,33,41,42,43) into a 1–12 urbanicity scale, most-urban → most-rural | Raw codes are grouped in tens (city=1x, suburb=2x, town=3x, rural=4x), so they are **not** linearly ordered as-is — a naive `LOCALE` cast to int is *not* usable for `polr`/`pmm` without this remap. NCES itself treats this as an ordinal urbanicity gradient. |
| `ADMCON7` | Remap: Neither required/recommended(3)=0 < Considered but not required(5)=1 < Recommended(2)=2 < Required(1)=3 | Raw numeric codes (1,2,3,4,5) are **not** in stringency order — do not cast directly. Code `4` ("Do not know") is not a stringency level at all — treat as missing (§2), not as a category. **[verified]** only codes `1`, `3`, `5` and missing actually occur in the current extract; `2` ("Recommended") and `4` ("Do not know") don't appear in this snapshot, but write the remap to handle all 5 dictionary codes rather than hardcoding to what's observed now. |

**`LOCALE2` — drop, don't harmonize.** The original draft of this doc assumed
`LOCALE2` (the legacy pre-2000s urbanicity coding) would need cross-walking
against `LOCALE` for older records. **[verified]** In
`Most-Recent-Cohorts-Institution.csv`, `LOCALE2` is `"NA"` for all 6,273 rows
— fully deprecated in this extract, no live values to harmonize. Just drop
the column. Re-check this if a different/historical extract is ever
substituted in, since `LOCALE2` was live in old-era data.

**Judgment calls — ordinal only under a restricted definition, flag for review:**

| Variable | Why it's not a clean single ordinal scale | Suggested treatment |
|---|---|---|
| `CCSIZSET` (Carnegie size & setting) | Conflates two-year size tiers, four-year size tiers, *and* residential intensity into one 18-level code — no single line orders "two-year, very large" against "four-year, small, highly residential." | Either (a) leave nominal for MICE (`polyreg`), or (b) engineer two derived ordinal features — an enrollment-size tier and a residential-intensity tier — and drop the composite. Recommend (b) if these fields matter to the analysis; (a) if they're low-priority covariates. |
| `CCUGPROF` (Carnegie undergrad profile) | Mixes two-year/four-year status with a selectivity/transfer-in ladder. Ordinal *within* the four-year subset, but not across the two-year/four-year boundary. | Same pattern as `CCSIZSET`: split into a selectivity-ladder ordinal (four-year rows only) plus reuse `ICLEVEL` for the two-year/four-year split, rather than treating all 15 codes as one line. |

---

## 2. Sentinel / structural codes — handle BEFORE building the ordinal scales above

This is the single easiest way to silently corrupt a MICE run on this data.
The dictionary defines two sentinel integers (`-1` "Not reported", `-2` "Not
applicable") across the Carnegie classification fields and `RELAFFIL` — but
**[verified]** what actually shows up in `Most-Recent-Cohorts-Institution.csv`
is *not* a clean 1:1 match to that, and the real behavior differs by field:

| Field | What's actually in the file | Handling |
|---|---|---|
| `CCBASIC`, `CCUGPROF`, `CCSIZSET` | Literal `-2` **does** appear (2,119 / 2,006 / 2,006 rows respectively) as a real structural sentinel, *and separately* the literal text `"NA"` appears (470 / 584 / 584 rows) for genuinely-missing cells. `-1` never appears at all in these three fields. | Treat exactly as originally planned: `-2` → split into a boolean "applicable" flag + exclude those rows from imputation of the substantive scale (don't impute a Carnegie classification onto an institution that structurally can't have one). `"NA"` text → real missing, becomes `NaN` (pandas does this automatically on read — see below) and *is* fair game for MICE to impute. |
| `RELAFFIL` | **Only** the literal text `"NA"` appears (5,409 rows) — `-1` and `-2` never occur as literal values anywhere in the column. | The not-reported vs. not-applicable distinction the dictionary implies for this field **is not recoverable from the field itself** in this extract — don't build an `RELAFFIL_APPLICABLE` flag expecting a `-2` to key off of; there isn't one. If that distinction matters, it has to be inferred indirectly (e.g. `CONTROL != 2` as a proxy for "not a private nonprofit, religious affiliation structurally unlikely"), otherwise just treat all `"NA"` as ordinary missing for MICE. |

Practical implication: reading this file with pandas defaults
(`pd.read_csv`) already converts the literal text `"NA"` to `NaN`
automatically — that part needs no special handling. The only thing that
needs an explicit step is **pulling `-2` back out before it corrupts the
substantive ordinal/nominal scale** — a `-2` sitting next to real codes
`0`–`33` will be read by `pmm`/`polr` as an extreme low value and distort the
imputation model for every other row if it's left in.

**Correction — no `"PrivacySuppressed"` in this file.** The original draft
flagged `"PrivacySuppressed"` (small-cell suppression) and `"NULL"` as
conventions to normalize. **[verified]** neither string occurs anywhere in
`Most-Recent-Cohorts-Institution.csv` (grepped the full file — zero matches).
That convention may still apply to *other* College Scorecard files (the
by-field-of-study or debt/repayment-detail extracts tend to carry it more),
so don't delete this check from the script, just don't expect it to fire on
this particular file. If a later `*_SUPP` companion column does show up in
some other extract used downstream, keep it as an auxiliary MICE predictor
rather than discarding it once the main value is nulled — suppression
correlates with small institution size, so it's MNAR, not MAR.

`OPEFLAG` has its own quirk: the dictionary lists a literal label `NA →
unknown` as one of its *actual categories* (Title-IV participation status
"unknown"), distinct from a blank/missing cell. **[verified]** in practice
this extract's `OPEFLAG` missing cells render the same as everywhere else
(plain `"NA"` on read), so if the "unknown" category needs to be
distinguished from true missing, that has to be sourced from elsewhere (e.g.
cross-referencing PEPS) — the column alone won't tell you which is which.

`OPENADMP` has a similar structural-vs-substantive split: value `3`
("does not enroll first-time students") means the open-admissions question
doesn't apply to that institution. **[verified]** value `3` doesn't actually
occur in this extract (only `1`, `2`, and missing do), but code the
`OPENADMP_APPLICABLE` split defensively anyway rather than assuming `3` will
never appear in a future pull.

---

## 3. Binary flags — already single columns, keep them that way

`MAIN`, `HBCU`, `PBI`, `ANNHI`, `TRIBAL`, `AANAPII`, `HSI`, `NANTI`,
`MENONLY`, `WOMENONLY`, `DISTANCEONLY`, `CURROPER`, `DOLPROVIDER`

**[verified]** all of these are single 0/1 columns (plus `"NA"` for missing
on a few, e.g. `DISTANCEONLY` and `DOLPROVIDER`) in
`Most-Recent-Cohorts-Institution.csv` — no `_0`/`_1` dummy-pair problem
exists in this file today. Leave them as one column each.

Keep this as a standing rule anyway for whatever step eventually does the
one-hot encoding of the nominal fields in §4: if that step is a generic
"one-hot every categorical column" utility rather than a hand-picked list, it
needs to either skip already-binary columns or call it with `drop_first=True`
— a 2-column dummy pair for something that's already binary is always
perfectly collinear (one column is `1 - other`) and will destabilize
regression-based imputation models (`logreg`, `norm`, `pmm`) for every other
variable that uses it as a predictor.

Also worth a note in the code: `CIP##ASSOC` / `CIP##BACHL` fields (program
offered indicators, one flag per 2-digit CIP family) look like a one-hot set
because there are ~30 of them, but they are **not** mutually exclusive
one-hot dummies of a single categorical variable — a school can and often
does offer programs in multiple CIP families simultaneously. Leave these as
independent binary flags; do not attempt to recombine them into a single
categorical "field of study" column.

---

## 4. Nominal (non-ordinal) categoricals — keep as dummies, but reduce cardinality first

These have no defensible single order. Leave them nominal for MICE
(`polyreg`/`lda` methods in R, or one-hot + a nominal-aware imputer in
Python), but a few need cardinality management before MICE, not after:

| Variable | Cardinality | Recommendation |
|---|---|---|
| `RELAFFIL` | ~80 denominations, plus `"NA"` for missing (**[verified]** no literal `-1`/`-2` in this extract — see §2) | Collapse to a handful of denomination families (Catholic / Mainline Protestant / Evangelical Protestant / Jewish / Muslim / Other / None) before one-hot + MICE. 80 sparse dummy columns for a single field is not worth the dimensionality cost. |
| `ACCREDAGENCY` / `ACCREDCODE` | Dozens of accreditors, string-valued | Do not one-hot. Frequency-encode or target-encode, or drop from the MICE predictor set entirely and only bring it back post-imputation if a downstream model needs it. |
| `ST_FIPS` | ~57 levels | Redundant with `REGION` (9 levels) for most modeling purposes. Recommend dropping `ST_FIPS` from the MICE variable set (or using it only as a grouping/clustering variable, not a one-hot predictor) and keeping `REGION`. |
| `CIPCODE1`–`CIPCODE6` | Hundreds of program codes, string-valued | Do not one-hot at the 6-digit level. If needed, roll up to 2-digit CIP family first. |
| `CONTROL` / `SCHTYPE` | 3 levels (Public / Private nonprofit / Private for-profit) | Genuinely nominal for most purposes — don't force an ordinal "public → nonprofit → for-profit" scale unless the specific analysis has a stated theoretical reason to (e.g. a commercial-orientation construct). Flag as an analyst judgment call if someone requests it. |
| `SCORECARD_SECTOR` | 15 levels | This is a deterministic function of `CONTROL` × `PREDDEG`(-ish). Recommend **dropping it entirely** before MICE — keeping it alongside its own inputs is redundant dimensionality and a source of perfect multicollinearity, not new information. Re-derive after imputation if a downstream table wants the combined sector label. |
| `REGION` | 9 levels, geographic | Nominal (no order), keep as dummies. |

---

## 5. Panel structure — only relevant if a multi-year file replaces this snapshot

**Correction:** `Most-Recent-Cohorts-Institution.csv` is a **cross-sectional
snapshot** — one row per institution, plain column names (`PREDDEG`,
`CCBASIC`, ...), 3,308 columns matching the dictionary 1:1. It is *not* the
30-cohort panel (1996–97 through 2025–26) that `Institution_cohort_map.csv`
describes — that would need year-suffixed columns or one row per
institution-year, and this file has neither. `LOCALE2` being 100% `"NA"`
(§1) is itself a symptom of this: it's a legacy field with no current value,
which only shows up as "empty" rather than "different by year" because
there's only one year here.

**If MICE is actually going to run on this single-year file as-is**, this
section doesn't apply — skip to §6. Confirm this with whoever scoped the
project before writing any of the code below.

**If a full multi-year panel gets substituted in later** (built from
`Institution_cohort_map.csv`'s variable list across all cohorts), the
following still needs to be handled and hasn't been verified against real
multi-year data yet:

- **`LOCALE` vs `LOCALE2`**: `LOCALE2` is the legacy pre-2000s urbanicity
  coding, `LOCALE` is the modern NCES scheme — not the same scale. A
  harmonized single locale variable would be needed across years, crosswalking
  `LOCALE2` into `LOCALE`'s categories for older cohorts (or vice versa).
- **`CCBASIC`/`CCUGPROF`/`CCSIZSET`**: Carnegie Classification is revised on
  its own periodic schedule (most recently 2018/2021), and category
  boundaries shifted between revisions — a code `15` in an early cohort isn't
  guaranteed to mean the same thing as `15` in a recent one. Would need
  either a single harmonized vintage or a classification-vintage/year
  indicator so MICE doesn't treat a definitional change as a real transition.
- Variables that only exist in a subset of years (fields introduced after a
  given IPEDS survey cycle change) are **structural** missingness-by-design,
  not something MICE should impute across — filtering each column to its
  actual valid-year range (per `Institution_cohort_map.csv`) would need to
  happen before imputation, not backfilling years where a field is
  definitionally absent for every institution.

---

## 6. Other pre-MICE steps worth doing at the same time

- **Predictor-matrix size.** With ~3,300 variables in the full dictionary,
  don't hand MICE a naive "everything predicts everything" matrix — it won't
  converge in reasonable time and small-cell categories will produce
  unstable/perfectly-separated logistic fits. Use `mice::quickpred()` (or an
  equivalent correlation/cardinality screen in Python) to build a reduced
  predictor matrix per target variable, and exclude the high-cardinality
  nominal fields from §4 as predictors for other variables even if you keep
  them as imputation targets.
- **Method assignment by type**, once §1–§4 are done: `pmm` for skewed
  continuous financial/earnings/debt fields (bounds imputed values to
  observed values — avoids impossible negative earnings that `norm` can
  produce), `logreg` for the cleaned single-column binary flags, `polr` for
  the ordinal variables from §1, `polyreg`/`lda` for the reduced-cardinality
  nominal variables from §4.
- **Drop near-zero-variance dummy levels** (e.g. a `RELAFFIL` denomination
  with a handful of institutions nationally) before MICE — they contribute
  almost no information and are a common source of perfect-separation
  warnings in the logistic imputation models MICE fits internally.
- **Re-expansion step.** Document that any variable collapsed to ordinal/
  single-binary for MICE gets re-expanded to dummies *after* imputation only
  if a specific downstream model requires one-hot inputs — don't re-expand
  by default, since it just reintroduces the collinearity problem this whole
  doc exists to avoid.
- **Post-imputation plausibility checks.** For the recombined ordinal fields,
  confirm imputed values round-trip to valid category codes (no imputed
  `LOCALE` rank between two real categories) and that `-2`/"not applicable"
  rows in §2 were correctly excluded rather than imputed.

---

## 7. Suggested build order for the preprocessing script

Written for `Most-Recent-Cohorts-Institution.csv` as it actually is
(cross-sectional, natively coded). The §0/§5 steps from the original draft
are kept as defensive checks, not because this file needs them today.

1. Load dictionary, build the `{VARIABLE NAME: [(VALUE, LABEL), ...]}` map to use as the ground-truth reference for steps below.
2. **[defensive, §0]** Confirm no column is actually one-hot encoded (guard against a future extract arriving pre-encoded); no-op on the current file.
3. Read the CSV with standard `NA`→`NaN` handling (pandas does this by default) — §2.
4. Pull out literal `-2` ("not applicable") from `CCBASIC`, `CCUGPROF`, `CCSIZSET` into per-column applicability flags, excluding those rows from imputation of the substantive scale — §2. (`RELAFFIL` has no recoverable `-2`; skip this step for it.)
5. Confirm the binary flags in §3 are still single columns (they are, today) before any generic categorical-encoding utility runs over the dataframe.
6. Encode the §1 ordinal variables directly as ordinal integers, applying the non-trivial remaps (`LOCALE`, `ADMCON7`) — never round-trip them through one-hot. Drop `LOCALE2` (100% `NA` in this file).
7. Reduce/collapse high-cardinality nominal fields (`RELAFFIL`, `ACCREDAGENCY`, `CIPCODE*`) — §4.
8. Drop redundant/derived fields (`SCORECARD_SECTOR`, `ST_FIPS` if `REGION` is retained) — §4.
9. One-hot encode only what's left in §4 (the genuinely nominal fields), with reduced cardinality already applied.
10. **[only if a multi-year panel replaces this snapshot — §5]** Harmonize schema-drift fields across cohorts and filter each column to its valid year range.
11. Build the reduced predictor matrix and per-variable method assignments — §6.
12. Run MICE; run the plausibility checks in §6 on the output.

---

## Open questions for whoever picks this up

- **Resolved 2026-08-28** by inspecting `Most-Recent-Cohorts-Institution.csv`
  directly: the data is natively coded, not one-hot encoded (see the
  correction at the top of this doc and the **[verified]** notes throughout).
  The original "what tool produced the one-hot columns" question no longer
  applies to this file.
- **New, from the same review:** is this single-year snapshot actually the
  file MICE will run on, or is it a stand-in for the fuller multi-year panel
  described by `Institution_cohort_map.csv`? §5's schema-drift/valid-year-range
  guidance only matters in the latter case — confirm which one this is before
  writing that part of the script.
- Is `ST_FIPS` needed downstream for anything `REGION` can't answer (e.g.
  state-specific policy variables)? If yes, keep it but exclude it from the
  MICE predictor matrix rather than dropping it outright.
- Which Carnegie Classification vintage does this extract's `CCBASIC` reflect,
  and — only relevant if a multi-year panel is built later — would
  harmonizing across vintages be in scope, or should that panel just be
  restricted to a single vintage's years?
- For `RELAFFIL`, does the not-reported vs. not-applicable distinction matter
  enough to the analysis to justify inferring it indirectly (e.g. via
  `CONTROL`), given the field itself no longer carries that distinction in
  this extract?


That output also confirms the 85% cutoff is doing real work at the margin — several columns sit just below the line (83.9%–84.9%), which is exactly the "some, most, or few variables might drop" sensitivity you flagged. Since the threshold is a single constant (COVERAGE_THRESHOLD at the top of the file), nudging it a couple points either way will visibly change how many of these borderline columns survive, without touching any other logic.

Summary of mice_pipeline.py (in college_scorecard_crosswalk/):

Coverage filter (COVERAGE_THRESHOLD = 0.85, toggle via APPLY_COVERAGE_FILTER) runs after the domain-specific structural fixes, not before — so a Carnegie field's coverage reflects real missingness (post -2 exclusion), not the inflated raw number.
Every domain rule from README_MICE.md is applied conditionally (if col in df.columns) — ordinal remaps (PREDDEG, HIGHDEG, ICLEVEL, LOCALE, ADMCON7), the -2/"not applicable" structural splits, OPENADMP's binary recode, RELAFFIL denomination collapsing, CIPCODE roll-up, and generic cardinality reduction for any high-cardinality nominal. None of it assumes a fixed variable list, so however many columns you end up pruning, nothing breaks.
Column classification is dynamic: registered ordinal/binary/nominal variables are claimed first, everything else falls back to numeric-vs-not (continuous vs. nominal) — robust to pandas dtype-naming changes (this is what caught the ACCREDAGENCY bug above).
run_mice() wraps miceforest (not installed in this environment — pip install miceforest is the only new dependency; everything else only needs pandas/numpy, which are present) and returns the M completed DataFrames per your last question, with ordinal columns rounded/clipped back to valid ranks after imputation since they're kept numeric (not categorical) specifically so LightGBM imputes them via regression and preserves order.
I ran everything short of the actual .mice() call against your real file to catch bugs before handing it off; that step still needs miceforest installed to execute. Want me to also wire in the ensembling step from our last conversation (fit-per-completed-dataset + average predictions) as a follow-on script, or leave that for when you're ready to plug in actual models?

