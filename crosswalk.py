"""
SEVIS School Code <-> IPEDS/OPE ID Crosswalk Builder
======================================================
Jupyter Notebook version — run each cell in order.

Before starting:
  1. Get a free College Scorecard API key at:
     https://api.data.gov/signup/
  2. Download the ICE Certified School List PDF from:
     https://studyinthestates.dhs.gov/school-search
     (click "Download Certified School List")
"""

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 1 — Install dependencies                                               │
# └─────────────────────────────────────────────────────────────────────────────┘

# %pip install requests rapidfuzz pdfplumber pandas tqdm


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 2 — Imports                                                            │
# └─────────────────────────────────────────────────────────────────────────────┘

import re
import requests
import pdfplumber
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm   # used only for the state-level matching loop


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 3 — Config  ← edit these before running                               │
# └─────────────────────────────────────────────────────────────────────────────┘

API_KEY      = "YOUR_KEY_HERE"
# Absolute path recommended for OneDrive/SharePoint environments
SEVP_PDF     = r"C:\Users\you\OneDrive\certified-school-list.pdf"
OUTPUT_CSV   = r"C:\Users\you\OneDrive\sevis_ipeds_crosswalk.csv"
SCORE_CUTOFF = 85   # 85 = good default; raise to 90 for precision, lower to 80 for recall


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 4 — Function definitions                                               │
# └─────────────────────────────────────────────────────────────────────────────┘

def fetch_scorecard(api_key: str) -> pd.DataFrame:
    """
    Fetches all institutions from the College Scorecard API.
    Returns a DataFrame with UNITID, OPE8_ID, OPE6_ID, name, city, state, zip.
    The API paginates at 100 records per page; loops until exhausted.
    """
    base_url = "https://api.data.gov/ed/collegescorecard/v1/schools"
    fields = ",".join([
        "id",           # UNITID
        "ope8_id",      # 8-digit OPE ID
        "ope6_id",      # 6-digit OPE ID
        "school.name",
        "school.city",
        "school.state",
        "school.zip",
    ])

    params = {
        "api_key": api_key,
        "fields":  fields,
        "per_page": 100,
        "page": 0,
    }

    records = []
    print("Fetching IPEDS data from College Scorecard API...")

    while True:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        records.extend(results)
        total   = data["metadata"]["total"]
        fetched = len(records)
        print(f"  {fetched:,}/{total:,} institutions fetched", end="\r")

        if fetched >= total:
            break
        params["page"] += 1

    print(f"\nDone — {len(records):,} institutions retrieved.")

    df = pd.DataFrame(records).rename(columns={
        "id":           "UNITID",
        "ope8_id":      "OPE8_ID",
        "ope6_id":      "OPE6_ID",
        "school.name":  "IPEDS_NAME",
        "school.city":  "IPEDS_CITY",
        "school.state": "IPEDS_STATE",
        "school.zip":   "IPEDS_ZIP",
    })

    # Zero-pad OPE IDs to standard widths
    df["OPE8_ID"] = df["OPE8_ID"].apply(
        lambda x: str(int(x)).zfill(8) if pd.notna(x) else None
    )
    df["OPE6_ID"] = df["OPE6_ID"].apply(
        lambda x: str(int(x)).zfill(6) if pd.notna(x) else None
    )

    return df


def parse_sevp_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Parses the ICE/SEVP Certified School List PDF.

    Expected columns (confirmed from studyinthestates.dhs.gov):
        SCHOOL NAME | CAMPUS NAME | F | M | CITY | ST | CAMPUS ID

    The Campus ID uniquely identifies each campus in SEVIS and is
    the numeric portion of a full SEVIS school code like BOS214F10096000.
    """
    rows = []
    header_pat = re.compile(
        r"SCHOOL\s+NAME.*CAMPUS\s+NAME.*CITY.*ST.*CAMPUS\s+ID",
        re.IGNORECASE,
    )

    print(f"Parsing SEVP PDF: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            if i % 50 == 0:
                print(f"  Page {i+1}/{n_pages}...", end="\r")
            table = page.extract_table()
            if table is None:
                continue
            for row in table:
                if row is None:
                    continue
                # Skip header rows
                joined = " ".join(str(c) for c in row if c)
                if header_pat.search(joined):
                    continue
                # Expect at least 7 columns
                if len(row) < 7:
                    continue
                school_name, campus_name, f_flag, m_flag, city, state, campus_id = (
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6]
                )
                if not school_name and not campus_name:
                    continue
                rows.append({
                    "SCHOOL_NAME": str(school_name or "").strip(),
                    "CAMPUS_NAME": str(campus_name or "").strip(),
                    "F_CERTIFIED": str(f_flag or "").strip(),
                    "M_CERTIFIED": str(m_flag or "").strip(),
                    "SEVP_CITY":   str(city or "").strip(),
                    "SEVP_STATE":  str(state or "").strip(),
                    "CAMPUS_ID":   str(campus_id or "").strip(),
                })

    df = pd.DataFrame(rows).drop_duplicates()
    print(f"Parsed {len(df):,} SEVP campus records.")
    return df


_REMOVE = re.compile(
    r"\b(university|univ|college|inst|institute|school|the|of|at|and|&|"
    r"inc|llc|corp|ltd|center|centre)\b",
    re.IGNORECASE,
)

def normalise(name: str) -> str:
    """Lowercase, strip punctuation, remove common stop-words."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = _REMOVE.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()


def normalise_zip(z) -> str:
    """Truncate to first 5 digits, handling 5-digit, hyphenated 9-digit,
    and run-together 9-digit formats."""
    if pd.isna(z) or str(z).strip() == "":
        return ""
    return str(z).split("-")[0].strip()[:5]


def zip_proximity(zip_a: str, zip_b: str, threshold: int = 5) -> str:
    """
    Compares two 5-digit zip strings and returns:
      "EXACT"   — identical
      "CLOSE"   — numeric difference <= threshold (default 5)
      "FAR"     — numeric difference > threshold
      "UNKNOWN" — one or both zips missing/non-numeric

    A threshold of 5 catches adjacent zip codes within the same
    postal district (e.g. a university with multiple zip codes
    for different buildings or campuses nearby).
    """
    if not zip_a or not zip_b:
        return "UNKNOWN"
    try:
        diff = abs(int(zip_a) - int(zip_b))
    except ValueError:
        return "UNKNOWN"
    if diff == 0:
        return "EXACT"
    if diff <= threshold:
        return "CLOSE"
    return "FAR"


def _empty_match() -> dict:
    return {
        "MATCH_SCORE":    None,
        "ZIP_PROXIMITY":  None,
        "UNITID":         None,
        "OPE8_ID":        None,
        "OPE6_ID":        None,
        "IPEDS_NAME":     None,
        "IPEDS_CITY":     None,
        "IPEDS_STATE":    None,
        "IPEDS_ZIP":      None,
    }


def _confidence(row) -> str:
    """
    Confidence tiers using name score + zip proximity:

      HIGH      — score >= 93 AND zip is EXACT
      MEDIUM    — score >= 93 AND zip is CLOSE or UNKNOWN
      MEDIUM    — score >= 85 AND zip is EXACT
      LOW       — score >= 93 AND zip is FAR
      LOW       — score >= 85 AND zip is CLOSE or UNKNOWN
      LOW       — score >= 85 AND zip is FAR
      UNMATCHED — no name match above cutoff
    """
    score    = row["MATCH_SCORE"]
    zip_prox = row["ZIP_PROXIMITY"]

    if pd.isna(score):
        return "UNMATCHED"
    if score >= 93 and zip_prox == "EXACT":
        return "HIGH"
    if score >= 93 and zip_prox in ("CLOSE", "UNKNOWN"):
        return "MEDIUM"
    if score >= 85 and zip_prox == "EXACT":
        return "MEDIUM"
    return "LOW"


def build_crosswalk(
    sevp: pd.DataFrame,
    ipeds: pd.DataFrame,
    score_cutoff: int = 85,
    zip_tiebreak_window: int = 5,
    zip_proximity_threshold: int = 5,
) -> pd.DataFrame:
    """
    Blocks on state, fuzzy-matches on name, then uses zip proximity as a
    tiebreaker when multiple candidates score within zip_tiebreak_window
    points of the top score.

    Parameters
    ----------
    score_cutoff : int
        Minimum token_set_ratio score to accept any match (default 85).
    zip_tiebreak_window : int
        Candidates within this many name-score points of the best are
        eligible for the zip tiebreak (default 5).
    zip_proximity_threshold : int
        Maximum numeric zip difference to be considered CLOSE (default 5).
    """
    ipeds = ipeds.copy()
    ipeds["_norm"] = ipeds["IPEDS_NAME"].fillna("").apply(normalise)
    ipeds["_zip5"] = ipeds["IPEDS_ZIP"].apply(normalise_zip)

    sevp = sevp.copy()
    sevp["_match_name"] = sevp.apply(
        lambda r: r["CAMPUS_NAME"] if r["CAMPUS_NAME"] else r["SCHOOL_NAME"],
        axis=1,
    )
    sevp["_norm"] = sevp["_match_name"].apply(normalise)
    sevp["_zip5"] = sevp["SEVP_ZIP"].apply(normalise_zip)

    results = []

    for state, sevp_group in tqdm(
        sevp.groupby("SEVP_STATE"), desc="States", unit="state"
    ):
        ipeds_state = ipeds[ipeds["IPEDS_STATE"] == state]
        if ipeds_state.empty:
            for _, row in sevp_group.iterrows():
                results.append({**row.to_dict(), **_empty_match()})
            continue

        candidates    = ipeds_state["_norm"].tolist()
        candidate_idx = ipeds_state.index.tolist()

        for _, row in sevp_group.iterrows():
            sevp_zip = row["_zip5"]

            # Step 1: get all candidates above cutoff
            all_matches = process.extract(
                row["_norm"],
                candidates,
                scorer=fuzz.token_set_ratio,
                score_cutoff=score_cutoff,
            )

            if not all_matches:
                results.append({**row.to_dict(), **_empty_match()})
                continue

            # Step 2: within the tiebreak window, prefer EXACT zip,
            # then CLOSE zip, then fall back to top name score
            best_score = all_matches[0][1]
            window = [
                (text, score, pos)
                for text, score, pos in all_matches
                if best_score - score <= zip_tiebreak_window
            ]

            exact_winner = None
            close_winner = None

            if sevp_zip:
                for text, score, pos in window:
                    ipeds_candidate = ipeds_state.loc[candidate_idx[pos]]
                    prox = zip_proximity(
                        sevp_zip,
                        ipeds_candidate["_zip5"],
                        threshold=zip_proximity_threshold,
                    )
                    if prox == "EXACT" and exact_winner is None:
                        exact_winner = (score, pos)
                    elif prox == "CLOSE" and close_winner is None:
                        close_winner = (score, pos)

            # Pick best winner in priority order
            if exact_winner is not None:
                final_score, final_pos = exact_winner
            elif close_winner is not None:
                final_score, final_pos = close_winner
            else:
                _, final_score, final_pos = all_matches[0]

            ipeds_row  = ipeds_state.loc[candidate_idx[final_pos]]
            zip_prox   = zip_proximity(
                sevp_zip,
                ipeds_row["_zip5"],
                threshold=zip_proximity_threshold,
            )

            results.append({
                **row.to_dict(),
                "MATCH_SCORE":   final_score,
                "ZIP_PROXIMITY": zip_prox,
                "UNITID":        ipeds_row["UNITID"],
                "OPE8_ID":       ipeds_row["OPE8_ID"],
                "OPE6_ID":       ipeds_row["OPE6_ID"],
                "IPEDS_NAME":    ipeds_row["IPEDS_NAME"],
                "IPEDS_CITY":    ipeds_row["IPEDS_CITY"],
                "IPEDS_STATE":   ipeds_row["IPEDS_STATE"],
                "IPEDS_ZIP":     ipeds_row["IPEDS_ZIP"],
            })

    out = pd.DataFrame(results).drop(
        columns=["_norm", "_match_name", "_zip5"], errors="ignore"
    )
    out["CONFIDENCE"] = out.apply(_confidence, axis=1)
    return out


def find_top_candidates(
    sevp_schools: pd.DataFrame,
    ipeds: pd.DataFrame,
    top_n: int = 10,
    score_cutoff: int = 60,
    zip_proximity_threshold: int = 5,
) -> pd.DataFrame:
    """
    For a list of hard-to-match SEVP schools, returns the top N
    College Scorecard candidates for each, with zip proximity scored.
    Searches nationally (no state blocking) since weak matches often
    fail because the state field itself is inconsistent.

    Parameters
    ----------
    sevp_schools : pd.DataFrame
        Subset of your sevp_df — the ~60 schools with no strong match.
        Must have the same columns as sevp_df.
    ipeds : pd.DataFrame
        Full IPEDS df from fetch_scorecard.
    top_n : int
        Number of candidates to return per SEVP school (default 10).
    score_cutoff : int
        Lower cutoff than build_crosswalk since we want more candidates
        for manual review (default 60).
    zip_proximity_threshold : int
        Passed to zip_proximity (default 5).
    """
    ipeds = ipeds.copy()
    ipeds["_norm"] = ipeds["IPEDS_NAME"].fillna("").apply(normalise)
    ipeds["_zip5"] = ipeds["IPEDS_ZIP"].apply(normalise_zip)

    sevp_schools = sevp_schools.copy()
    sevp_schools["_match_name"] = sevp_schools.apply(
        lambda r: r["CAMPUS_NAME"] if r["CAMPUS_NAME"] else r["SCHOOL_NAME"],
        axis=1,
    )
    sevp_schools["_norm"] = sevp_schools["_match_name"].apply(normalise)
    sevp_schools["_zip5"] = sevp_schools["SEVP_ZIP"].apply(normalise_zip)

    all_candidates    = ipeds["_norm"].tolist()
    all_candidate_idx = ipeds.index.tolist()

    rows = []
    for _, sevp_row in sevp_schools.iterrows():
        sevp_zip = sevp_row["_zip5"]

        matches = process.extract(
            sevp_row["_norm"],
            all_candidates,
            scorer=fuzz.token_set_ratio,
            score_cutoff=score_cutoff,
            limit=top_n,
        )

        if not matches:
            # Still write one row so the school appears in output
            rows.append({
                "SCHOOL_NAME":   sevp_row["SCHOOL_NAME"],
                "CAMPUS_NAME":   sevp_row["CAMPUS_NAME"],
                "CAMPUS_ID":     sevp_row["CAMPUS_ID"],
                "SEVP_CITY":     sevp_row["SEVP_CITY"],
                "SEVP_STATE":    sevp_row["SEVP_STATE"],
                "SEVP_ZIP":      sevp_row.get("SEVP_ZIP", ""),
                "CANDIDATE_RANK": None,
                "MATCH_SCORE":   None,
                "ZIP_PROXIMITY": None,
                "UNITID":        None,
                "OPE8_ID":       None,
                "OPE6_ID":       None,
                "IPEDS_NAME":    None,
                "IPEDS_CITY":    None,
                "IPEDS_STATE":   None,
                "IPEDS_ZIP":     None,
            })
            continue

        for rank, (_, score, pos) in enumerate(matches, start=1):
            ipeds_row = ipeds.loc[all_candidate_idx[pos]]
            zip_prox  = zip_proximity(
                sevp_zip,
                ipeds_row["_zip5"],
                threshold=zip_proximity_threshold,
            )
            rows.append({
                "SCHOOL_NAME":    sevp_row["SCHOOL_NAME"],
                "CAMPUS_NAME":    sevp_row["CAMPUS_NAME"],
                "CAMPUS_ID":      sevp_row["CAMPUS_ID"],
                "SEVP_CITY":      sevp_row["SEVP_CITY"],
                "SEVP_STATE":     sevp_row["SEVP_STATE"],
                "SEVP_ZIP":       sevp_row.get("SEVP_ZIP", ""),
                "CANDIDATE_RANK": rank,
                "MATCH_SCORE":    score,
                "ZIP_PROXIMITY":  zip_prox,
                "UNITID":         ipeds_row["UNITID"],
                "OPE8_ID":        ipeds_row["OPE8_ID"],
                "OPE6_ID":        ipeds_row["OPE6_ID"],
                "IPEDS_NAME":     ipeds_row["IPEDS_NAME"],
                "IPEDS_CITY":     ipeds_row["IPEDS_CITY"],
                "IPEDS_STATE":    ipeds_row["IPEDS_STATE"],
                "IPEDS_ZIP":      ipeds_row["IPEDS_ZIP"],
            })

    out = pd.DataFrame(rows).drop(
        columns=["_norm", "_match_name", "_zip5"], errors="ignore"
    )
    return out


def print_summary(df: pd.DataFrame) -> None:
    total  = len(df)
    counts = df["CONFIDENCE"].value_counts()
    print("\n── Match Summary ──────────────────────────────")
    for tier in ["HIGH", "MEDIUM", "LOW", "UNMATCHED"]:
        n   = counts.get(tier, 0)
        pct = 100 * n / total if total else 0
        print(f"  {tier:<12}: {n:>6,}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<12}: {total:>6,}")
    print("───────────────────────────────────────────────\n")


print("Functions defined.")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 5 — Fetch IPEDS data (cached)                                          │
# └─────────────────────────────────────────────────────────────────────────────┘

IPEDS_CACHE = r"ipeds.csv"   # change path if you want it saved elsewhere

from pathlib import Path

if Path(IPEDS_CACHE).exists():
    ipeds_df = pd.read_csv(IPEDS_CACHE, dtype={"OPE8_ID": str, "OPE6_ID": str,
                                                "UNITID": str, "IPEDS_ZIP": str})
    print(f"Loaded IPEDS from cache → {IPEDS_CACHE} ({len(ipeds_df):,} rows)")
else:
    ipeds_df = fetch_scorecard(API_KEY)
    ipeds_df.to_csv(IPEDS_CACHE, index=False)
    print(f"IPEDS saved to cache → {IPEDS_CACHE}")

ipeds_df.head()


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 6 — Parse SEVP PDF (cached)                                            │
# └─────────────────────────────────────────────────────────────────────────────┘

SEVP_CACHE = r"sevp.csv"   # change path if you want it saved elsewhere

if Path(SEVP_CACHE).exists():
    sevp_df = pd.read_csv(SEVP_CACHE, dtype={"CAMPUS_ID": str, "SEVP_ZIP": str})
    print(f"Loaded SEVP from cache → {SEVP_CACHE} ({len(sevp_df):,} rows)")
else:
    sevp_df = parse_sevp_pdf(SEVP_PDF)
    sevp_df.to_csv(SEVP_CACHE, index=False)
    print(f"SEVP saved to cache → {SEVP_CACHE}")

sevp_df.head()


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 7 — Build crosswalk                                                    │
# └─────────────────────────────────────────────────────────────────────────────┘

crosswalk = build_crosswalk(sevp_df, ipeds_df, score_cutoff=SCORE_CUTOFF)
print_summary(crosswalk)


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 8 — Inspect results interactively                                      │
# └─────────────────────────────────────────────────────────────────────────────┘

SAMPLE_COLS = ["SCHOOL_NAME", "IPEDS_NAME", "SEVP_STATE",
               "SEVP_CITY",   "IPEDS_CITY", "OPE6_ID",
               "UNITID",      "MATCH_SCORE", "CONFIDENCE"]

print("── HIGH confidence sample ──")
display(crosswalk[crosswalk["CONFIDENCE"] == "HIGH"][SAMPLE_COLS].head(10))

print("── MEDIUM confidence sample (spot-check these) ──")
display(crosswalk[crosswalk["CONFIDENCE"] == "MEDIUM"][SAMPLE_COLS].head(10))

print("── UNMATCHED sample (SEVIS-only schools, e.g. language schools) ──")
display(crosswalk[crosswalk["CONFIDENCE"] == "UNMATCHED"][SAMPLE_COLS].head(10))


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 9 — Save outputs                                                       │
# └─────────────────────────────────────────────────────────────────────────────┘

# Full crosswalk
crosswalk.to_csv(OUTPUT_CSV, index=False)
print(f"Full crosswalk saved → {OUTPUT_CSV}")

# Separate review file for LOW / UNMATCHED records
review_csv = OUTPUT_CSV.replace(".csv", "_review.csv")
review = crosswalk[crosswalk["CONFIDENCE"].isin(["LOW", "UNMATCHED"])]
review.to_csv(review_csv, index=False)
print(f"Review file ({len(review):,} records) saved → {review_csv}")


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 10 — Top-N candidates for hard-to-match schools                        │
# └─────────────────────────────────────────────────────────────────────────────┘

# Build your list of hard-to-match schools — either pass in CAMPUS_IDs:
HARD_TO_MATCH_IDS = [
    "10096000",  # replace with your actual CAMPUS_IDs
    "12345678",
    # ...
]

# Filter sevp_df to just those schools
sevp_hard = sevp_df[sevp_df["CAMPUS_ID"].isin(HARD_TO_MATCH_IDS)].copy()

# Or alternatively filter from your unmatched/low crosswalk rows:
# sevp_hard = sevp_df[sevp_df["CAMPUS_ID"].isin(
#     crosswalk[crosswalk["CONFIDENCE"].isin(["LOW", "UNMATCHED"])]["CAMPUS_ID"]
# )].copy()

print("Running top-N candidate search for " + str(len(sevp_hard)) + " schools...")
candidates_df = find_top_candidates(
    sevp_schools=sevp_hard,
    ipeds=ipeds_df,
    top_n=10,
    score_cutoff=60,          # lower than main crosswalk to surface more options
    zip_proximity_threshold=5,
)

# Save
candidates_csv = OUTPUT_CSV.replace(".csv", "_top_candidates.csv")
candidates_df.to_csv(candidates_csv, index=False)
print("Top candidates saved → " + candidates_csv)

# Preview — each SEVP school gets up to 10 rows, sorted by rank
display(candidates_df.head(30))