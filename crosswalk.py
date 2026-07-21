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

# %pip install requests rapidfuzz pdfplumber pandas


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 2 — Imports                                                            │
# └─────────────────────────────────────────────────────────────────────────────┘

import math
import re
import requests
import pdfplumber
import pandas as pd
from rapidfuzz import fuzz, process


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 3 — Config  ← edit these before running                               │
# └─────────────────────────────────────────────────────────────────────────────┘

API_KEY      = "YOUR_KEY_HERE"
# Absolute path recommended for OneDrive/SharePoint environments
SEVP_PDF     = r"C:\Users\you\OneDrive\certified-school-list.pdf"
OUTPUT_CSV   = r"C:\Users\you\OneDrive\sevis_ipeds_crosswalk.csv"
SCORE_CUTOFF = 85   # 85 = good default; raise to 90 for precision, lower to 80 for recall
EXACT_MILES  = 1    # <= this many miles apart counts as EXACT for LOCATION_PROXIMITY (same campus/building)
CLOSE_MILES  = 10   # <= this many miles apart counts as CLOSE (same metro area); farther is FAR


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 4 — Function definitions                                               │
# └─────────────────────────────────────────────────────────────────────────────┘

def fetch_scorecard(api_key: str) -> pd.DataFrame:
    """
    Fetches all institutions from the College Scorecard API.
    Returns a DataFrame with UNITID, OPE8_ID, OPE6_ID, name, city, state, zip,
    and latitude/longitude (used for geographic fuzzy-match tiebreaking).
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
        "location.lat",
        "location.lon",
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
        "location.lat": "IPEDS_LAT",
        "location.lon": "IPEDS_LON",
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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/long points."""
    EARTH_RADIUS_MI = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * math.asin(math.sqrt(a))


def location_proximity(
    lat_a, lon_a, lat_b, lon_b,
    exact_miles: float = 1.0,
    close_miles: float = 10.0,
):
    """
    Compares two lat/long points and returns a (tier, distance_miles) tuple:
      "EXACT"   — <= exact_miles apart (default 1 mi — same campus/building)
      "CLOSE"   — <= close_miles apart (default 10 mi — same metro area)
      "FAR"     — farther than close_miles
      "UNKNOWN" — one or both coordinates missing/non-numeric

    Sliding scale: distance_miles is returned alongside the tier so the
    exact gap is visible in the output, not just a bucket label.
    """
    try:
        lat_a, lon_a, lat_b, lon_b = float(lat_a), float(lon_a), float(lat_b), float(lon_b)
    except (TypeError, ValueError):
        return "UNKNOWN", None
    if any(pd.isna(v) for v in (lat_a, lon_a, lat_b, lon_b)):
        return "UNKNOWN", None

    dist = haversine_distance(lat_a, lon_a, lat_b, lon_b)
    if dist <= exact_miles:
        return "EXACT", dist
    if dist <= close_miles:
        return "CLOSE", dist
    return "FAR", dist


def _empty_match() -> dict:
    return {
        "MATCH_SCORE":         None,
        "LOCATION_PROXIMITY":  None,
        "DISTANCE_MILES":      None,
        "UNITID":              None,
        "OPE8_ID":             None,
        "OPE6_ID":             None,
        "IPEDS_NAME":          None,
        "IPEDS_CITY":          None,
        "IPEDS_STATE":         None,
        "IPEDS_LAT":           None,
        "IPEDS_LON":           None,
    }


def _confidence(row) -> str:
    """
    Confidence tiers using name score + geographic proximity:

      HIGH      — score >= 93 AND location is EXACT
      MEDIUM    — score >= 93 AND location is CLOSE or UNKNOWN
      MEDIUM    — score >= 85 AND location is EXACT
      LOW       — score >= 93 AND location is FAR
      LOW       — score >= 85 AND location is CLOSE, FAR, or UNKNOWN
      UNMATCHED — no name match above cutoff
    """
    score    = row["MATCH_SCORE"]
    loc_prox = row["LOCATION_PROXIMITY"]

    if pd.isna(score):
        return "UNMATCHED"
    if score >= 93 and loc_prox == "EXACT":
        return "HIGH"
    if score >= 93 and loc_prox in ("CLOSE", "UNKNOWN"):
        return "MEDIUM"
    if score >= 85 and loc_prox == "EXACT":
        return "MEDIUM"
    return "LOW"


def build_crosswalk(
    sevp: pd.DataFrame,
    ipeds: pd.DataFrame,
    score_cutoff: int = 85,
    score_tiebreak_window: int = 5,
    exact_miles: float = 1.0,
    close_miles: float = 10.0,
) -> pd.DataFrame:
    """
    Blocks on state, fuzzy-matches on name, then uses geographic distance
    (haversine, in miles) as a tiebreaker when multiple candidates score
    within score_tiebreak_window points of the top name-match score.

    Requires SEVP_LAT/SEVP_LON columns on `sevp` and IPEDS_LAT/IPEDS_LON
    columns on `ipeds`.

    Parameters
    ----------
    score_cutoff : int
        Minimum token_set_ratio score to accept any match (default 85).
    score_tiebreak_window : int
        Candidates within this many name-score points of the best are
        eligible for the distance tiebreak (default 5).
    exact_miles : float
        Distance at or under which two points count as EXACT (default 1).
    close_miles : float
        Distance at or under which two points count as CLOSE (default 10).
    """
    ipeds = ipeds.copy()
    ipeds["_norm"] = ipeds["IPEDS_NAME"].fillna("").apply(normalise)
    ipeds["IPEDS_LAT"] = pd.to_numeric(ipeds["IPEDS_LAT"], errors="coerce")
    ipeds["IPEDS_LON"] = pd.to_numeric(ipeds["IPEDS_LON"], errors="coerce")

    sevp = sevp.copy()
    sevp["_match_name"] = sevp.apply(
        lambda r: r["CAMPUS_NAME"] if r["CAMPUS_NAME"] else r["SCHOOL_NAME"],
        axis=1,
    )
    sevp["_norm"] = sevp["_match_name"].apply(normalise)
    sevp["SEVP_LAT"] = pd.to_numeric(sevp["SEVP_LAT"], errors="coerce")
    sevp["SEVP_LON"] = pd.to_numeric(sevp["SEVP_LON"], errors="coerce")

    results = []
    state_groups = sevp.groupby("SEVP_STATE")
    n_states = sevp["SEVP_STATE"].nunique()

    for i, (state, sevp_group) in enumerate(state_groups, start=1):
        print(f"  Matching state {i}/{n_states}: {state}...", end="\r")

        ipeds_state = ipeds[ipeds["IPEDS_STATE"] == state]
        if ipeds_state.empty:
            for _, row in sevp_group.iterrows():
                results.append({**row.to_dict(), **_empty_match()})
            continue

        candidates    = ipeds_state["_norm"].tolist()
        candidate_idx = ipeds_state.index.tolist()

        for _, row in sevp_group.iterrows():
            sevp_lat, sevp_lon = row["SEVP_LAT"], row["SEVP_LON"]

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

            # Step 2: within the tiebreak window, prefer EXACT distance,
            # then CLOSE distance, then fall back to top name score
            best_score = all_matches[0][1]
            window = [
                (text, score, pos)
                for text, score, pos in all_matches
                if best_score - score <= score_tiebreak_window
            ]

            exact_winner = None
            close_winner = None

            for _text, score, pos in window:
                ipeds_candidate = ipeds_state.loc[candidate_idx[pos]]
                prox, _dist = location_proximity(
                    sevp_lat, sevp_lon,
                    ipeds_candidate["IPEDS_LAT"], ipeds_candidate["IPEDS_LON"],
                    exact_miles=exact_miles, close_miles=close_miles,
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

            ipeds_row = ipeds_state.loc[candidate_idx[final_pos]]
            loc_prox, distance_miles = location_proximity(
                sevp_lat, sevp_lon,
                ipeds_row["IPEDS_LAT"], ipeds_row["IPEDS_LON"],
                exact_miles=exact_miles, close_miles=close_miles,
            )

            results.append({
                **row.to_dict(),
                "MATCH_SCORE":        final_score,
                "LOCATION_PROXIMITY": loc_prox,
                "DISTANCE_MILES":     round(distance_miles, 2) if distance_miles is not None else None,
                "UNITID":             ipeds_row["UNITID"],
                "OPE8_ID":            ipeds_row["OPE8_ID"],
                "OPE6_ID":            ipeds_row["OPE6_ID"],
                "IPEDS_NAME":         ipeds_row["IPEDS_NAME"],
                "IPEDS_CITY":         ipeds_row["IPEDS_CITY"],
                "IPEDS_STATE":        ipeds_row["IPEDS_STATE"],
                "IPEDS_LAT":          ipeds_row["IPEDS_LAT"],
                "IPEDS_LON":          ipeds_row["IPEDS_LON"],
            })

    print(f"  Matched {n_states}/{n_states} states.        ")

    out = pd.DataFrame(results).drop(
        columns=["_norm", "_match_name"], errors="ignore"
    )
    out["CONFIDENCE"] = out.apply(_confidence, axis=1)
    return out


def find_top_candidates(
    sevp_schools: pd.DataFrame,
    ipeds: pd.DataFrame,
    top_n: int = 10,
    score_cutoff: int = 60,
    exact_miles: float = 1.0,
    close_miles: float = 10.0,
) -> pd.DataFrame:
    """
    For a list of hard-to-match SEVP schools, returns the top N
    College Scorecard candidates for each, with geographic distance scored.
    Searches nationally (no state blocking) since weak matches often
    fail because the state field itself is inconsistent.

    Requires SEVP_LAT/SEVP_LON columns on `sevp_schools` and IPEDS_LAT/
    IPEDS_LON columns on `ipeds`.

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
    exact_miles, close_miles : float
        Passed to location_proximity (defaults 1 and 10).
    """
    ipeds = ipeds.copy()
    ipeds["_norm"] = ipeds["IPEDS_NAME"].fillna("").apply(normalise)
    ipeds["IPEDS_LAT"] = pd.to_numeric(ipeds["IPEDS_LAT"], errors="coerce")
    ipeds["IPEDS_LON"] = pd.to_numeric(ipeds["IPEDS_LON"], errors="coerce")

    sevp_schools = sevp_schools.copy()
    sevp_schools["_match_name"] = sevp_schools.apply(
        lambda r: r["CAMPUS_NAME"] if r["CAMPUS_NAME"] else r["SCHOOL_NAME"],
        axis=1,
    )
    sevp_schools["_norm"] = sevp_schools["_match_name"].apply(normalise)
    sevp_schools["SEVP_LAT"] = pd.to_numeric(sevp_schools["SEVP_LAT"], errors="coerce")
    sevp_schools["SEVP_LON"] = pd.to_numeric(sevp_schools["SEVP_LON"], errors="coerce")

    all_candidates    = ipeds["_norm"].tolist()
    all_candidate_idx = ipeds.index.tolist()

    rows = []
    for _, sevp_row in sevp_schools.iterrows():
        sevp_lat, sevp_lon = sevp_row["SEVP_LAT"], sevp_row["SEVP_LON"]

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
                "SCHOOL_NAME":         sevp_row["SCHOOL_NAME"],
                "CAMPUS_NAME":         sevp_row["CAMPUS_NAME"],
                "CAMPUS_ID":           sevp_row["CAMPUS_ID"],
                "SEVP_CITY":           sevp_row["SEVP_CITY"],
                "SEVP_STATE":          sevp_row["SEVP_STATE"],
                "SEVP_LAT":            sevp_row.get("SEVP_LAT"),
                "SEVP_LON":            sevp_row.get("SEVP_LON"),
                "CANDIDATE_RANK":      None,
                "MATCH_SCORE":         None,
                "LOCATION_PROXIMITY":  None,
                "DISTANCE_MILES":      None,
                "UNITID":              None,
                "OPE8_ID":             None,
                "OPE6_ID":             None,
                "IPEDS_NAME":          None,
                "IPEDS_CITY":          None,
                "IPEDS_STATE":         None,
                "IPEDS_LAT":           None,
                "IPEDS_LON":           None,
            })
            continue

        for rank, (_, score, pos) in enumerate(matches, start=1):
            ipeds_row = ipeds.loc[all_candidate_idx[pos]]
            loc_prox, distance_miles = location_proximity(
                sevp_lat, sevp_lon,
                ipeds_row["IPEDS_LAT"], ipeds_row["IPEDS_LON"],
                exact_miles=exact_miles, close_miles=close_miles,
            )
            rows.append({
                "SCHOOL_NAME":         sevp_row["SCHOOL_NAME"],
                "CAMPUS_NAME":         sevp_row["CAMPUS_NAME"],
                "CAMPUS_ID":           sevp_row["CAMPUS_ID"],
                "SEVP_CITY":           sevp_row["SEVP_CITY"],
                "SEVP_STATE":          sevp_row["SEVP_STATE"],
                "SEVP_LAT":            sevp_row.get("SEVP_LAT"),
                "SEVP_LON":            sevp_row.get("SEVP_LON"),
                "CANDIDATE_RANK":      rank,
                "MATCH_SCORE":         score,
                "LOCATION_PROXIMITY":  loc_prox,
                "DISTANCE_MILES":      round(distance_miles, 2) if distance_miles is not None else None,
                "UNITID":              ipeds_row["UNITID"],
                "OPE8_ID":             ipeds_row["OPE8_ID"],
                "OPE6_ID":             ipeds_row["OPE6_ID"],
                "IPEDS_NAME":          ipeds_row["IPEDS_NAME"],
                "IPEDS_CITY":          ipeds_row["IPEDS_CITY"],
                "IPEDS_STATE":         ipeds_row["IPEDS_STATE"],
                "IPEDS_LAT":           ipeds_row["IPEDS_LAT"],
                "IPEDS_LON":           ipeds_row["IPEDS_LON"],
            })

    out = pd.DataFrame(rows).drop(
        columns=["_norm", "_match_name"], errors="ignore"
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

# NOTE: if you have an older cached ipeds.csv without IPEDS_LAT/IPEDS_LON,
# delete it (or add those two columns yourself) before running this cell —
# build_crosswalk requires them.
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

# NOTE: parse_sevp_pdf() does not produce SEVP_LAT/SEVP_LON — add those two
# columns to sevp.csv yourself (e.g. by geocoding SEVP_CITY/SEVP_STATE, or
# a campus address) before running Cell 7. build_crosswalk requires them.
if Path(SEVP_CACHE).exists():
    sevp_df = pd.read_csv(SEVP_CACHE, dtype={"CAMPUS_ID": str})
    print(f"Loaded SEVP from cache → {SEVP_CACHE} ({len(sevp_df):,} rows)")
else:
    sevp_df = parse_sevp_pdf(SEVP_PDF)
    sevp_df.to_csv(SEVP_CACHE, index=False)
    print(f"SEVP saved to cache → {SEVP_CACHE}")

sevp_df.head()


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 7 — Build crosswalk                                                    │
# └─────────────────────────────────────────────────────────────────────────────┘

crosswalk = build_crosswalk(
    sevp_df, ipeds_df,
    score_cutoff=SCORE_CUTOFF,
    exact_miles=EXACT_MILES,
    close_miles=CLOSE_MILES,
)
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
    exact_miles=EXACT_MILES,
    close_miles=CLOSE_MILES,
)

# Save
candidates_csv = OUTPUT_CSV.replace(".csv", "_top_candidates.csv")
candidates_df.to_csv(candidates_csv, index=False)
print("Top candidates saved → " + candidates_csv)

# Preview — each SEVP school gets up to 10 rows, sorted by rank
display(candidates_df.head(30))