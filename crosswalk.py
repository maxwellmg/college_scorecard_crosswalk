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
SCORE_CUTOFF   = 85    # min NAME_SCORE (token_set_ratio) for a candidate to be considered at all
MAX_MILES      = 25    # LATLON_SCORE slides from 100 (0 mi apart) to 0 (>= this many miles apart)
ZIP_MAX_DIFF   = 500   # ZIP_SCORE slides from 100 (identical) to 0 (>= this numeric zip difference)
HIGH_CUTOFF    = 90    # OVERALL_SCORE (avg of NAME/CITY/ZIP/LATLON scores) >= this → CONFIDENCE = HIGH
MEDIUM_CUTOFF  = 75    # OVERALL_SCORE >= this (but < HIGH_CUTOFF) → CONFIDENCE = MEDIUM; below → LOW


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


def normalise_zip(z) -> str:
    """Truncate to first 5 digits, handling 5-digit, hyphenated 9-digit,
    and run-together 9-digit formats."""
    if pd.isna(z) or str(z).strip() == "":
        return ""
    return str(z).split("-")[0].strip()[:5]


def normalise_city(city) -> str:
    """Lowercase, strip punctuation, collapse whitespace. No stop-word
    removal — city names don't carry the institutional noise normalise()
    strips (\"university\", \"college\", etc.)."""
    if pd.isna(city) or str(city).strip() == "":
        return ""
    city = str(city).lower()
    city = re.sub(r"[^a-z0-9\s]", " ", city)
    return re.sub(r"\s+", " ", city).strip()


def city_score(city_a, city_b):
    """
    Fuzzy token_set_ratio between two city names, 0-100.
    Returns None if either side is missing/blank — excluded from the
    OVERALL_SCORE average rather than counted as a mismatch.
    """
    ca, cb = normalise_city(city_a), normalise_city(city_b)
    if not ca or not cb:
        return None
    return float(fuzz.token_set_ratio(ca, cb))


def zip_score(zip_a, zip_b, max_diff: float = 500.0):
    """
    Compares two 5-digit zips (after normalise_zip) and returns a
    (label, score) tuple:
      label — "Y" (exact match), "N" (different), "NA" (one/both missing)
      score — slides linearly from 100 (identical) down to 0 (numeric
      difference >= max_diff), same shape as latlon_score's decay.

    Zip codes aren't uniformly spaced geographically, so a numeric
    difference is only a rough regional-proximity proxy, not a distance —
    nearby zips are usually (not always) nearby places.

    None is excluded from the OVERALL_SCORE average rather than counted
    as a failure — an unverifiable zip shouldn't drag the match down.
    """
    za, zb = normalise_zip(zip_a), normalise_zip(zip_b)
    if not za or not zb:
        return "NA", None
    try:
        diff = abs(int(za) - int(zb))
    except ValueError:
        return "NA", None
    label = "Y" if diff == 0 else "N"
    score = max(0.0, 100.0 * (1 - diff / max_diff))
    return label, score


def latlon_score(lat_a, lon_a, lat_b, lon_b, max_miles: float = 25.0):
    """
    Compares two lat/long points and returns a (distance_miles, score)
    tuple. Score slides linearly from 100 (0 miles apart) down to 0
    (>= max_miles apart) — a continuous read on "how well does location
    corroborate this match" rather than a fixed EXACT/CLOSE/FAR bucket.

    Returns (None, None) if either coordinate is missing/non-numeric —
    excluded from the OVERALL_SCORE average rather than counted as a
    failure.
    """
    try:
        lat_a, lon_a, lat_b, lon_b = float(lat_a), float(lon_a), float(lat_b), float(lon_b)
    except (TypeError, ValueError):
        return None, None
    if any(pd.isna(v) for v in (lat_a, lon_a, lat_b, lon_b)):
        return None, None

    dist = haversine_distance(lat_a, lon_a, lat_b, lon_b)
    score = max(0.0, 100.0 * (1 - dist / max_miles))
    return dist, score


def _score_candidate(name_score: float, sevp_row, ipeds_row, max_miles: float, zip_max_diff: float = 500.0) -> dict:
    """
    Computes the four component scores (name/city/zip/latlon) for one
    SEVP-row / IPEDS-candidate pair, plus OVERALL_SCORE — the mean of
    whichever components are actually available. A component that can't
    be verified (missing data on either side) is left out of the average
    entirely rather than counted as 0, so schools missing a zip or a
    geocode aren't penalised for it.
    """
    city_sc              = city_score(sevp_row.get("SEVP_CITY"), ipeds_row["IPEDS_CITY"])
    zip_label, zip_sc     = zip_score(sevp_row.get("SEVP_ZIP"), ipeds_row["IPEDS_ZIP"], max_diff=zip_max_diff)
    distance_mi, latlon_sc = latlon_score(
        sevp_row.get("SEVP_LAT"), sevp_row.get("SEVP_LON"),
        ipeds_row["IPEDS_LAT"], ipeds_row["IPEDS_LON"],
        max_miles=max_miles,
    )

    components = [s for s in (name_score, city_sc, zip_sc, latlon_sc) if s is not None]
    overall = sum(components) / len(components) if components else None

    return {
        "NAME_SCORE":     round(name_score, 2) if name_score is not None else None,
        "CITY_SCORE":     round(city_sc, 2) if city_sc is not None else None,
        "ZIP_SCORE":      zip_sc,
        "ZIP_MATCH":      zip_label,
        "LATLON_SCORE":   round(latlon_sc, 2) if latlon_sc is not None else None,
        "DISTANCE_MILES": round(distance_mi, 2) if distance_mi is not None else None,
        "OVERALL_SCORE":  round(overall, 2) if overall is not None else None,
    }


def _empty_match() -> dict:
    return {
        "NAME_SCORE":      None,
        "CITY_SCORE":      None,
        "ZIP_SCORE":       None,
        "ZIP_MATCH":       None,
        "LATLON_SCORE":    None,
        "DISTANCE_MILES":  None,
        "OVERALL_SCORE":   None,
        "UNITID":          None,
        "OPE8_ID":         None,
        "OPE6_ID":         None,
        "IPEDS_NAME":      None,
        "IPEDS_CITY":      None,
        "IPEDS_STATE":     None,
        "IPEDS_ZIP":       None,
        "IPEDS_LAT":       None,
        "IPEDS_LON":       None,
    }


def _confidence(row, high_cutoff: float = 90, medium_cutoff: float = 75) -> str:
    """
    Confidence tiers off OVERALL_SCORE — the average of NAME_SCORE,
    CITY_SCORE, ZIP_SCORE, and LATLON_SCORE (NA components excluded):

      HIGH      — OVERALL_SCORE >= high_cutoff
      MEDIUM    — OVERALL_SCORE >= medium_cutoff
      LOW       — OVERALL_SCORE below medium_cutoff
      UNMATCHED — no name candidate above SCORE_CUTOFF at all
    """
    score = row["OVERALL_SCORE"]
    if pd.isna(score):
        return "UNMATCHED"
    if score >= high_cutoff:
        return "HIGH"
    if score >= medium_cutoff:
        return "MEDIUM"
    return "LOW"


def build_crosswalk(
    sevp: pd.DataFrame,
    ipeds: pd.DataFrame,
    score_cutoff: int = 85,
    max_miles: float = 25.0,
    zip_max_diff: float = 500.0,
    high_cutoff: float = 90,
    medium_cutoff: float = 75,
) -> pd.DataFrame:
    """
    Blocks on state, then for every SEVP row: shortlists IPEDS candidates
    by name (token_set_ratio >= score_cutoff), scores each candidate on
    four independent signals — NAME_SCORE, CITY_SCORE, ZIP_SCORE,
    LATLON_SCORE — and picks whichever candidate has the highest
    OVERALL_SCORE (the mean of whatever signals are actually available;
    see _score_candidate). CONFIDENCE is bucketed off OVERALL_SCORE.

    Requires SEVP_LAT/SEVP_LON columns on `sevp` and IPEDS_LAT/IPEDS_LON
    columns on `ipeds`. SEVP_ZIP on `sevp` is optional but recommended —
    without it, ZIP_SCORE is NA for every row and OVERALL_SCORE falls
    back to averaging name/city/latlon only.

    Parameters
    ----------
    score_cutoff : int
        Minimum NAME_SCORE (token_set_ratio) for a candidate to be
        considered at all (default 85). This is a name-only gate — it
        keeps the candidate shortlist sane before the other three signals
        get a vote; a candidate below this never reaches the composite.
    max_miles : float
        Distance at which LATLON_SCORE bottoms out at 0 (default 25).
    zip_max_diff : float
        Numeric zip difference at which ZIP_SCORE bottoms out at 0
        (default 500).
    high_cutoff, medium_cutoff : float
        OVERALL_SCORE thresholds for CONFIDENCE (defaults 90 and 75).
    """
    ipeds = ipeds.copy()
    ipeds["_norm"] = ipeds["IPEDS_NAME"].fillna("").apply(normalise)
    ipeds["IPEDS_LAT"] = pd.to_numeric(ipeds["IPEDS_LAT"], errors="coerce")
    ipeds["IPEDS_LON"] = pd.to_numeric(ipeds["IPEDS_LON"], errors="coerce")

    sevp = sevp.copy()
    sevp["_match_name"] = sevp.apply(
        # SCHOOL_NAME (the registered institution name) is the primary match
        # target — it's the direct counterpart to IPEDS_NAME. CAMPUS_NAME is
        # often just a branch/location label (e.g. a bare city name), and
        # matching on that alone can produce spurious high name scores
        # against unrelated schools that happen to share that location name.
        # Branch-campus disambiguation is handled by CITY_SCORE/LATLON_SCORE
        # instead, so CAMPUS_NAME is only a fallback when SCHOOL_NAME is blank.
        lambda r: r["SCHOOL_NAME"] if r["SCHOOL_NAME"] else r["CAMPUS_NAME"],
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
            # Step 1: shortlist every candidate above the name cutoff
            # (limit=None — don't silently cap at rapidfuzz's default of 5)
            all_matches = process.extract(
                row["_norm"],
                candidates,
                scorer=fuzz.token_set_ratio,
                score_cutoff=score_cutoff,
                limit=None,
            )

            if not all_matches:
                results.append({**row.to_dict(), **_empty_match()})
                continue

            # Step 2: score every shortlisted candidate on all 4 signals
            # and keep whichever has the best OVERALL_SCORE (ties go to
            # the higher NAME_SCORE)
            scored = []
            for _text, name_score, pos in all_matches:
                ipeds_candidate = ipeds_state.loc[candidate_idx[pos]]
                scored.append((
                    _score_candidate(name_score, row, ipeds_candidate, max_miles, zip_max_diff=zip_max_diff),
                    ipeds_candidate,
                ))

            best_scores, best_ipeds_row = max(
                scored, key=lambda c: (c[0]["OVERALL_SCORE"], c[0]["NAME_SCORE"])
            )

            results.append({
                **row.to_dict(),
                **best_scores,
                "UNITID":      best_ipeds_row["UNITID"],
                "OPE8_ID":     best_ipeds_row["OPE8_ID"],
                "OPE6_ID":     best_ipeds_row["OPE6_ID"],
                "IPEDS_NAME":  best_ipeds_row["IPEDS_NAME"],
                "IPEDS_CITY":  best_ipeds_row["IPEDS_CITY"],
                "IPEDS_STATE": best_ipeds_row["IPEDS_STATE"],
                "IPEDS_ZIP":   best_ipeds_row["IPEDS_ZIP"],
                "IPEDS_LAT":   best_ipeds_row["IPEDS_LAT"],
                "IPEDS_LON":   best_ipeds_row["IPEDS_LON"],
            })

    print(f"  Matched {n_states}/{n_states} states.        ")

    out = pd.DataFrame(results).drop(
        columns=["_norm", "_match_name"], errors="ignore"
    )
    out["CONFIDENCE"] = out.apply(
        lambda r: _confidence(r, high_cutoff=high_cutoff, medium_cutoff=medium_cutoff),
        axis=1,
    )
    return out


def find_top_candidates(
    sevp_schools: pd.DataFrame,
    ipeds: pd.DataFrame,
    top_n: int = 10,
    score_cutoff: int = 60,
    max_miles: float = 25.0,
    zip_max_diff: float = 500.0,
) -> pd.DataFrame:
    """
    For a list of hard-to-match SEVP schools, returns the top N College
    Scorecard candidates for each, scored the same way as build_crosswalk
    (NAME_SCORE, CITY_SCORE, ZIP_SCORE, LATLON_SCORE, OVERALL_SCORE).
    Candidates are shortlisted by name (score_cutoff) then ranked by
    OVERALL_SCORE — CANDIDATE_RANK 1 is the best combined-evidence match,
    not necessarily the best name match alone.
    Searches nationally (no state blocking) since weak matches often
    fail because the state field itself is inconsistent.

    Requires SEVP_LAT/SEVP_LON columns on `sevp_schools` and IPEDS_LAT/
    IPEDS_LON columns on `ipeds`. SEVP_ZIP is optional (see build_crosswalk).

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
    max_miles : float
        Passed to latlon_score (default 25).
    zip_max_diff : float
        Passed to zip_score (default 500).
    """
    ipeds = ipeds.copy()
    ipeds["_norm"] = ipeds["IPEDS_NAME"].fillna("").apply(normalise)
    ipeds["IPEDS_LAT"] = pd.to_numeric(ipeds["IPEDS_LAT"], errors="coerce")
    ipeds["IPEDS_LON"] = pd.to_numeric(ipeds["IPEDS_LON"], errors="coerce")

    sevp_schools = sevp_schools.copy()
    sevp_schools["_match_name"] = sevp_schools.apply(
        # SCHOOL_NAME (the registered institution name) is the primary match
        # target — it's the direct counterpart to IPEDS_NAME. CAMPUS_NAME is
        # often just a branch/location label (e.g. a bare city name), and
        # matching on that alone can produce spurious high name scores
        # against unrelated schools that happen to share that location name.
        # Branch-campus disambiguation is handled by CITY_SCORE/LATLON_SCORE
        # instead, so CAMPUS_NAME is only a fallback when SCHOOL_NAME is blank.
        lambda r: r["SCHOOL_NAME"] if r["SCHOOL_NAME"] else r["CAMPUS_NAME"],
        axis=1,
    )
    sevp_schools["_norm"] = sevp_schools["_match_name"].apply(normalise)
    sevp_schools["SEVP_LAT"] = pd.to_numeric(sevp_schools["SEVP_LAT"], errors="coerce")
    sevp_schools["SEVP_LON"] = pd.to_numeric(sevp_schools["SEVP_LON"], errors="coerce")

    all_candidates    = ipeds["_norm"].tolist()
    all_candidate_idx = ipeds.index.tolist()

    rows = []
    for _, sevp_row in sevp_schools.iterrows():
        matches = process.extract(
            sevp_row["_norm"],
            all_candidates,
            scorer=fuzz.token_set_ratio,
            score_cutoff=score_cutoff,
            limit=None,
        )

        if not matches:
            # Still write one row so the school appears in output
            rows.append({
                "SCHOOL_NAME":    sevp_row["SCHOOL_NAME"],
                "CAMPUS_NAME":    sevp_row["CAMPUS_NAME"],
                "CAMPUS_ID":      sevp_row["CAMPUS_ID"],
                "SEVP_CITY":      sevp_row["SEVP_CITY"],
                "SEVP_STATE":     sevp_row["SEVP_STATE"],
                "SEVP_LAT":       sevp_row.get("SEVP_LAT"),
                "SEVP_LON":       sevp_row.get("SEVP_LON"),
                "CANDIDATE_RANK": None,
                **_empty_match(),
            })
            continue

        scored = []
        for _text, name_score, pos in matches:
            ipeds_row = ipeds.loc[all_candidate_idx[pos]]
            scored.append((
                _score_candidate(name_score, sevp_row, ipeds_row, max_miles, zip_max_diff=zip_max_diff),
                ipeds_row,
            ))

        # Rank by combined evidence, not raw name score
        scored.sort(key=lambda c: (c[0]["OVERALL_SCORE"], c[0]["NAME_SCORE"]), reverse=True)

        for rank, (scores, ipeds_row) in enumerate(scored[:top_n], start=1):
            rows.append({
                "SCHOOL_NAME":    sevp_row["SCHOOL_NAME"],
                "CAMPUS_NAME":    sevp_row["CAMPUS_NAME"],
                "CAMPUS_ID":      sevp_row["CAMPUS_ID"],
                "SEVP_CITY":      sevp_row["SEVP_CITY"],
                "SEVP_STATE":     sevp_row["SEVP_STATE"],
                "SEVP_LAT":       sevp_row.get("SEVP_LAT"),
                "SEVP_LON":       sevp_row.get("SEVP_LON"),
                "CANDIDATE_RANK": rank,
                **scores,
                "UNITID":      ipeds_row["UNITID"],
                "OPE8_ID":     ipeds_row["OPE8_ID"],
                "OPE6_ID":     ipeds_row["OPE6_ID"],
                "IPEDS_NAME":  ipeds_row["IPEDS_NAME"],
                "IPEDS_CITY":  ipeds_row["IPEDS_CITY"],
                "IPEDS_STATE": ipeds_row["IPEDS_STATE"],
                "IPEDS_ZIP":   ipeds_row["IPEDS_ZIP"],
                "IPEDS_LAT":   ipeds_row["IPEDS_LAT"],
                "IPEDS_LON":   ipeds_row["IPEDS_LON"],
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
# A SEVP_ZIP column is optional but recommended — if present, it powers the
# ZIP_MATCH Y/N sanity-check column (compared against IPEDS_ZIP).
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

crosswalk = build_crosswalk(
    sevp_df, ipeds_df,
    score_cutoff=SCORE_CUTOFF,
    max_miles=MAX_MILES,
    zip_max_diff=ZIP_MAX_DIFF,
    high_cutoff=HIGH_CUTOFF,
    medium_cutoff=MEDIUM_CUTOFF,
)
print_summary(crosswalk)


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CELL 8 — Inspect results interactively                                      │
# └─────────────────────────────────────────────────────────────────────────────┘

SAMPLE_COLS = ["SCHOOL_NAME", "IPEDS_NAME", "SEVP_STATE",
               "SEVP_CITY",   "IPEDS_CITY", "OPE6_ID",
               "UNITID",      "NAME_SCORE", "CITY_SCORE",
               "ZIP_SCORE",   "LATLON_SCORE", "OVERALL_SCORE",
               "CONFIDENCE"]

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
    max_miles=MAX_MILES,
    zip_max_diff=ZIP_MAX_DIFF,
)

# Save
candidates_csv = OUTPUT_CSV.replace(".csv", "_top_candidates.csv")
candidates_df.to_csv(candidates_csv, index=False)
print("Top candidates saved → " + candidates_csv)

# Preview — each SEVP school gets up to 10 rows, sorted by rank
display(candidates_df.head(30))