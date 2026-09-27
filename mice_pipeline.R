# =============================================================================
# College Scorecard - MICE preprocessing + imputation (R port)
# =============================================================================
# Single-file R port of mice_pipeline.py, built specifically to avoid the
# LightGBM/miceforest native-crash class documented in
# HANDOFF_MICE_CRASH_DEBUG.md. Uses R's `mice` package (parametric methods:
# pmm / logreg / polyreg) instead of a tree-based/LightGBM approach, so that
# entire crash class does not apply here.
#
# Mirrors mice_pipeline.py's structure and domain logic 1:1 (same variable
# registries, same ordinal remaps, same sentinel handling, same coverage/
# degeneracy filters) so it can be cross-referenced against README_MICE.md
# and the Python source directly. See that repo's README_MICE.md for the
# domain rationale behind every rule below - this file only re-implements
# the mechanics, in R, as a single self-contained script.
#
# Dependencies: base R plus the `mice` package only.
#   install.packages("mice")
#
# Usage: edit the config block below, then either source() this file or
# run it with Rscript mice_pipeline.R
# =============================================================================

suppressMessages(library(mice))

# -----------------------------------------------------------------------
# Config - edit these for your run
# -----------------------------------------------------------------------

RAW_CSV        <- "Most-Recent-Cohorts-Institution.csv"   # <-- edit
DICTIONARY_CSV <- "CollegeScorecardDataDictionary.csv"    # <-- edit
OUTPUT_DIR     <- "mice_output_r"

COVERAGE_THRESHOLD <- 0.85   # min non-missing share to keep a column
N_DATASETS   <- 5            # m  - number of completed datasets MICE produces
N_ITERATIONS <- 5            # maxit
RANDOM_STATE <- 42

APPLY_COVERAGE_FILTER <- TRUE

# See mice_pipeline.py's MIN_CATEGORY_COUNT/MIN_CATEGORY_FRACTION docstring
# for why both an absolute and a relative floor exist.
MIN_CATEGORY_COUNT    <- 20
MIN_CATEGORY_FRACTION <- 0.01
MAX_DOMINANT_VALUE_SHARE <- 0.99

METADATA_COLUMN_SUFFIXES <- c("year_added")

PROTECTED_FROM_VARIANCE_DROP <- c(
  "HBCU", "PBI", "ANNHI", "TRIBAL", "AANAPII", "HSI", "NANTI",
  "MENONLY", "WOMENONLY"
)

# -----------------------------------------------------------------------
# Domain registries - from CollegeScorecardDataDictionary.csv, matching
# mice_pipeline.py's registries of the same names.
# -----------------------------------------------------------------------

ALWAYS_EXCLUDE <- c(
  "OPEID", "OPEID6", "INSTNM", "CITY", "STABBR", "ZIP",
  "INSTURL", "NPCURL", "ACCREDCODE",
  "ST_FIPS", "SCORECARD_SECTOR", "LOCALE2"
)

# var -> raw code meaning "structurally not applicable" (not missing).
STRUCTURAL_NA <- list(CCBASIC = -2, CCUGPROF = -2, CCSIZSET = -2, OPENADMP = 3)

# var -> ordered vector of raw codes, low -> high, or NULL meaning "sort the
# observed codes ascending". Kept as plain numeric (not an ordered factor)
# and imputed via pmm, same rationale as the Python side: pmm on the numeric
# rank respects order and is far more numerically robust than polr at this
# column count, and the values get rounded/clipped back to a valid rank
# after imputation (see run_mice_r).
ORDINAL_ORDER <- list(
  PREDDEG = NULL, HIGHDEG = NULL, ICLEVEL = NULL, LOCALE = NULL,
  ADMCON7 = c(3, 5, 2, 1)
)
ORDINAL_MISSING_CODES <- list(ADMCON7 = c(4))

BINARY_COLS <- c(
  "MAIN", "HBCU", "PBI", "ANNHI", "TRIBAL", "AANAPII", "HSI", "NANTI",
  "MENONLY", "WOMENONLY", "DISTANCEONLY", "CURROPER", "DOLPROVIDER"
)

NOMINAL_COLS <- c("CONTROL", "REGION", "SCHTYPE", "OPEFLAG")

CIP_COLS <- paste0("CIPCODE", 1:6)

# RELAFFIL: ~80 denominations collapsed to families. Not exhaustive by
# construction - anything observed but not listed here falls to "Other"
# rather than becoming spuriously missing (see collapse_relaffil).
RELAFFIL_FAMILY_MAP <- c(
  "30" = "Catholic", "80" = "Jewish", "106" = "Muslim", "94" = "Latter Day Saints",
  "91" = "Orthodox Christian", "92" = "Orthodox Christian", "110" = "Orthodox Christian",
  "93" = "Other", "65" = "Other",
  "42" = "Interdenominational", "78" = "Interdenominational",
  "108" = "Non-Denominational", "88" = "Non-Denominational",
  "22" = "Mainline Protestant", "39" = "Mainline Protestant", "53" = "Mainline Protestant",
  "66" = "Mainline Protestant", "67" = "Mainline Protestant", "68" = "Mainline Protestant",
  "71" = "Mainline Protestant", "73" = "Mainline Protestant", "76" = "Mainline Protestant",
  "50" = "Mainline Protestant", "60" = "Mainline Protestant", "61" = "Mainline Protestant",
  "97" = "Mainline Protestant", "103" = "Mainline Protestant",
  "24" = "Evangelical Protestant", "27" = "Evangelical Protestant", "28" = "Evangelical Protestant",
  "33" = "Evangelical Protestant", "34" = "Evangelical Protestant", "35" = "Evangelical Protestant",
  "36" = "Evangelical Protestant", "37" = "Evangelical Protestant", "38" = "Evangelical Protestant",
  "40" = "Evangelical Protestant", "41" = "Evangelical Protestant", "43" = "Evangelical Protestant",
  "44" = "Evangelical Protestant", "45" = "Evangelical Protestant", "47" = "Evangelical Protestant",
  "48" = "Evangelical Protestant", "49" = "Evangelical Protestant", "51" = "Evangelical Protestant",
  "52" = "Evangelical Protestant", "54" = "Evangelical Protestant", "55" = "Evangelical Protestant",
  "57" = "Evangelical Protestant", "58" = "Evangelical Protestant", "59" = "Evangelical Protestant",
  "64" = "Evangelical Protestant", "69" = "Evangelical Protestant", "74" = "Evangelical Protestant",
  "75" = "Evangelical Protestant", "77" = "Evangelical Protestant", "79" = "Evangelical Protestant",
  "81" = "Evangelical Protestant", "84" = "Evangelical Protestant", "87" = "Evangelical Protestant",
  "89" = "Evangelical Protestant", "95" = "Evangelical Protestant", "99" = "Evangelical Protestant",
  "100" = "Evangelical Protestant", "101" = "Evangelical Protestant", "102" = "Evangelical Protestant",
  "105" = "Evangelical Protestant", "107" = "Evangelical Protestant"
)

# -----------------------------------------------------------------------
# Step 0 - naming convention crosswalk (dotted API names <-> flat VARIABLE NAME)
# -----------------------------------------------------------------------

build_dotted_to_flat <- function(dictionary_csv) {
  dict <- read.csv(dictionary_csv, stringsAsFactors = FALSE, check.names = FALSE)
  colnames(dict)[1] <- sub("^﻿", "", colnames(dict)[1])

  dotted_to_flat <- character(0)
  seen <- character(0)
  var_col <- dict[["VARIABLE NAME"]]
  friendly_col <- dict[["developer-friendly name"]]
  category_col <- dict[["dev-category"]]

  for (i in seq_len(nrow(dict))) {
    var <- trimws(var_col[i])
    if (is.na(var) || var == "" || var %in% seen) next
    seen[length(seen) + 1] <- var
    friendly <- trimws(friendly_col[i])
    if (is.na(friendly) || friendly == "") next
    category <- trimws(category_col[i])
    dotted <- if (is.na(category) || category %in% c("", "root")) friendly else paste0(category, ".", friendly)
    dotted_to_flat[[dotted]] <- var
  }
  dotted_to_flat
}

rename_to_flat_convention <- function(df, dotted_to_flat) {
  cols <- colnames(df)
  matched <- cols %in% names(dotted_to_flat)
  colnames(df)[matched] <- unname(dotted_to_flat[cols[matched]])
  list(df = df, n_renamed = sum(matched))
}

# -----------------------------------------------------------------------
# Step 1 - load
# -----------------------------------------------------------------------

load_raw <- function(path, dictionary_csv) {
  df <- read.csv(
    path,
    na.strings = c("NA", "PrivacySuppressed", "NULL", "PS", ""),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  colnames(df)[1] <- sub("^﻿", "", colnames(df)[1])

  dotted_to_flat <- build_dotted_to_flat(dictionary_csv)
  renamed <- rename_to_flat_convention(df, dotted_to_flat)
  df <- renamed$df
  cat(sprintf(
    "Renamed %d dotted-convention column(s) to the flat convention (0 is expected/harmless if the input file is already flat-named).\n",
    renamed$n_renamed
  ))

  if ("UNITID" %in% colnames(df)) {
    rownames(df) <- as.character(df$UNITID)
    df$UNITID <- NULL
  }
  df <- df[, !(colnames(df) %in% ALWAYS_EXCLUDE), drop = FALSE]
  df
}

# -----------------------------------------------------------------------
# Step 1b - set aside per-variable metadata columns, then checkpoint sparsity
# -----------------------------------------------------------------------

split_metadata_columns <- function(df, suffixes = METADATA_COLUMN_SUFFIXES) {
  cols_lower <- tolower(colnames(df))
  pattern <- paste0("(", paste(tolower(suffixes), collapse = "|"), ")$")
  is_metadata <- grepl(pattern, cols_lower)
  list(
    mice_df = df[, !is_metadata, drop = FALSE],
    metadata_df = df[, is_metadata, drop = FALSE],
    metadata_cols = colnames(df)[is_metadata]
  )
}

report_sparsity <- function(df) {
  coverage <- vapply(df, function(col) mean(!is.na(col)), numeric(1))
  out <- data.frame(column = names(coverage), coverage = as.numeric(coverage), row.names = NULL)
  out[order(out$coverage), ]
}

# -----------------------------------------------------------------------
# Step 2 - domain-specific structural fixes (run BEFORE coverage filtering)
# -----------------------------------------------------------------------

apply_structural_na <- function(df) {
  for (var in names(STRUCTURAL_NA)) {
    if (!(var %in% colnames(df))) next
    sentinel <- STRUCTURAL_NA[[var]]
    is_missing <- is.na(df[[var]])
    applicable <- ifelse(is_missing, NA, as.numeric(df[[var]] != sentinel))
    df[[paste0(var, "_APPLICABLE")]] <- applicable
    sentinel_rows <- !is_missing & df[[var]] == sentinel
    df[[var]][sentinel_rows] <- NA
  }
  df
}

recode_openadmp <- function(df) {
  if ("OPENADMP" %in% colnames(df)) {
    df$OPENADMP <- ifelse(df$OPENADMP == 1, 1, ifelse(df$OPENADMP == 2, 0, NA))
  }
  df
}

collapse_relaffil <- function(df) {
  if (!("RELAFFIL" %in% colnames(df))) return(df)
  raw <- as.character(df$RELAFFIL)
  mapped <- unname(RELAFFIL_FAMILY_MAP[raw])
  unmapped_but_present <- is.na(mapped) & !is.na(df$RELAFFIL)
  mapped[unmapped_but_present] <- "Other"
  df$RELAFFIL <- mapped
  df
}

encode_ordinal <- function(df) {
  rank_maps <- list()
  for (var in names(ORDINAL_ORDER)) {
    if (!(var %in% colnames(df))) next
    order <- ORDINAL_ORDER[[var]]
    missing_codes <- ORDINAL_MISSING_CODES[[var]]
    working <- df[[var]]
    if (!is.null(missing_codes)) working[working %in% missing_codes] <- NA
    categories <- if (is.null(order)) sort(unique(working[!is.na(working)])) else order
    rank_map <- setNames(seq_along(categories) - 1, as.character(categories))
    df[[var]] <- unname(rank_map[as.character(working)])
    rank_maps[[var]] <- rank_map
  }
  list(df = df, rank_maps = rank_maps)
}

prepare_cip_cols <- function(df) {
  for (col in CIP_COLS) {
    if (!(col %in% colnames(df))) next
    vals <- as.character(df[[col]])
    df[[col]] <- ifelse(is.na(vals), NA, substr(vals, 1, 2))
  }
  df
}

reduce_cardinality <- function(x, max_levels = 15, min_freq = 0.01) {
  counts <- table(x, useNA = "no")
  if (length(counts) <= max_levels) return(x)
  share <- sort(counts / sum(counts), decreasing = TRUE)
  keep_n <- min(max_levels, sum(share >= min_freq))
  keep <- names(share)[seq_len(keep_n)]
  ifelse(is.na(x) | as.character(x) %in% keep, x, "Other")
}

apply_structural_fixes <- function(df) {
  df <- apply_structural_na(df)
  df <- recode_openadmp(df)
  df <- collapse_relaffil(df)
  df <- prepare_cip_cols(df)
  res <- encode_ordinal(df)
  list(df = res$df, rank_maps = res$rank_maps)
}

# -----------------------------------------------------------------------
# Step 3 - coverage-based column selection ("the magic proportion")
# -----------------------------------------------------------------------

select_by_coverage <- function(df, threshold) {
  coverage <- vapply(df, function(col) mean(!is.na(col)), numeric(1))
  coverage <- sort(coverage)
  dropped <- coverage[coverage < threshold]
  kept <- names(coverage)[coverage >= threshold]
  list(df = df[, kept, drop = FALSE], dropped = dropped)
}

# -----------------------------------------------------------------------
# Step 3b - degenerate / near-constant column dropping
# -----------------------------------------------------------------------

drop_degenerate_columns <- function(df, candidate_cols, min_category_count = MIN_CATEGORY_COUNT,
                                     min_category_fraction = MIN_CATEGORY_FRACTION) {
  dropped <- character(0)
  n_rows <- nrow(df)
  required <- max(min_category_count, min_category_fraction * n_rows)
  for (col in candidate_cols) {
    if (col %in% PROTECTED_FROM_VARIANCE_DROP) next
    counts <- sort(table(df[[col]], useNA = "no"), decreasing = TRUE)
    if (length(counts) < 2) {
      dropped <- c(dropped, col)
      next
    }
    if (counts[2] < required) dropped <- c(dropped, col)
  }
  list(df = df[, !(colnames(df) %in% dropped), drop = FALSE], dropped = dropped)
}

drop_near_constant_continuous <- function(df, candidate_cols, max_share = MAX_DOMINANT_VALUE_SHARE) {
  dropped <- character(0)
  for (col in candidate_cols) {
    if (col %in% PROTECTED_FROM_VARIANCE_DROP) next
    x <- df[[col]][!is.na(df[[col]])]
    if (length(x) == 0) {
      dropped <- c(dropped, col)
      next
    }
    top_share <- max(table(x)) / length(x)
    if (top_share >= max_share) dropped <- c(dropped, col)
  }
  list(df = df[, !(colnames(df) %in% dropped), drop = FALSE], dropped = dropped)
}

# -----------------------------------------------------------------------
# Step 4 - classify whatever columns survived, dynamically; finalize types
# -----------------------------------------------------------------------

classify_columns <- function(df) {
  ordinal_cols <- intersect(names(ORDINAL_ORDER), colnames(df))
  applicable_flags <- intersect(paste0(names(STRUCTURAL_NA), "_APPLICABLE"), colnames(df))
  binary_cols <- intersect(BINARY_COLS, colnames(df))
  if ("OPENADMP" %in% colnames(df)) binary_cols <- c(binary_cols, "OPENADMP")
  binary_cols <- c(binary_cols, applicable_flags)
  nominal_cols <- intersect(NOMINAL_COLS, colnames(df))

  claimed <- unique(c(ordinal_cols, binary_cols, nominal_cols))
  remaining <- setdiff(colnames(df), claimed)

  is_num <- vapply(df[remaining], is.numeric, logical(1))
  continuous_cols <- remaining[is_num]
  nominal_cols <- c(nominal_cols, remaining[!is_num])

  list(ordinal = ordinal_cols, binary = binary_cols, nominal = nominal_cols, continuous = continuous_cols)
}

finalize_types <- function(df, columns) {
  for (col in columns$nominal) {
    df[[col]] <- reduce_cardinality(df[[col]])
  }
  for (col in c(columns$ordinal, columns$continuous)) {
    df[[col]] <- as.numeric(df[[col]])
  }
  for (col in columns$binary) {
    df[[col]] <- factor(df[[col]])
  }
  for (col in columns$nominal) {
    df[[col]] <- factor(df[[col]])
  }
  df
}

# -----------------------------------------------------------------------
# Step 5 - run MICE
# -----------------------------------------------------------------------

build_method_vector <- function(df, columns) {
  method <- setNames(rep("", ncol(df)), colnames(df))
  method[columns$ordinal] <- "pmm"
  method[columns$continuous] <- "pmm"
  method[columns$binary] <- "logreg"
  method[columns$nominal] <- "polyreg"
  method
}

run_mice_r <- function(df, columns, rank_maps) {
  method_vec <- build_method_vector(df, columns)

  # Standardize every numeric (ordinal + continuous) column before handing
  # anything to mice. Confirmed against real data: Scorecard mixes raw
  # dollar amounts (tens of thousands) with 0-1 proportions in the SAME
  # regression, which spans many orders of magnitude in xtx and breaks
  # floating-point conditioning even where the predictors aren't actually
  # collinear - ridge and rank-based predictor pruning alone kept hitting
  # "computationally singular" on the exact same fingerprint regardless of
  # which predictors were removed, which is the signature of a scale
  # problem, not a true-rank problem. Same principle post_mice_modeling.py
  # already applies downstream (its StandardScaler step) - applying it
  # here too, before imputation, keeps both stages consistent. Means/sds
  # are saved to unscale the completed datasets back to their real units.
  numeric_cols <- c(columns$ordinal, columns$continuous)
  scale_center <- setNames(rep(0, length(numeric_cols)), numeric_cols)
  scale_sd <- setNames(rep(1, length(numeric_cols)), numeric_cols)
  for (col in numeric_cols) {
    x <- df[[col]]
    m <- mean(x, na.rm = TRUE)
    s <- stats::sd(x, na.rm = TRUE)
    if (is.na(s) || s == 0) s <- 1
    scale_center[col] <- m
    scale_sd[col] <- s
    df[[col]] <- (x - m) / s
  }

  # Same predictor-matrix-size concern as README_MICE.md section 6 flags for
  # the Python side - quickpred keeps the predictor set manageable rather
  # than handing mice() a naive "everything predicts everything" matrix.
  #
  # mincor=0.3 (not quickpred's usual 0.1 default) is deliberate: confirmed
  # against real data, mincor=0.1 lets in enough of Scorecard's near-
  # duplicate variable pairs (a raw rate and its "_POOLED" 2-year-rolling-
  # average twin are correlated well above 0.9) that the predictor matrix
  # for some targets becomes exactly collinear - "system is computationally
  # singular" with a reciprocal condition number around 1e-24, which no
  # reasonable ridge value papers over. Raising mincor keeps a predictor
  # only when it's more informative about the target, which thins out
  # exactly this kind of redundant pair.
  pred_matrix <- mice::quickpred(df, mincor = 0.3)

  # Even at mincor=0.3, a predictor SET can still be exactly collinear with
  # ITSELF (mincor only filters correlation with the target, not among the
  # chosen predictors) - confirmed against real data with a reciprocal
  # condition number around 1e-19, i.e. genuine rank deficiency, not just
  # "highly correlated." Scorecard has several variable families that are
  # linear identities by construction (e.g. income-bracket percentages that
  # sum to 100% across a fixed set of columns) - selecting several members
  # of such a family as predictors together is exactly collinear, and no
  # ridge value fixes that (see the ridge note below - the penalty scales
  # down proportionally to the same near-zero pivot it's supposed to fix).
  #
  # Fix: build each target's predictor set GREEDILY, ranked by correlation
  # with the target, but skip any candidate that's highly correlated
  # (> MAX_PREDICTOR_INTERCORRELATION) with a predictor ALREADY chosen for
  # that target. This is what actually breaks up "sums-to-100%" and
  # "_POOLED near-duplicate" families, which a flat top-N cut by
  # correlation-with-target alone does not (it can still pick several
  # members of the same collinear family, just the top few).
  N_MAX_PREDICTORS <- 25
  MAX_PREDICTOR_INTERCORRELATION <- 0.9
  suppressWarnings(cor_matrix <- abs(stats::cor(
    data.matrix(df), use = "pairwise.complete.obs"
  )))
  cor_matrix[is.na(cor_matrix)] <- 0

  # Mean/mode-filled stand-in for what mice's OWN "working data" looks like
  # mid-run: by the time it regresses on a given target, every OTHER
  # variable has already been filled with its current best-guess imputed
  # value (from an earlier iteration or the initial random draw), so the
  # predictor columns mice actually regresses against have no missing
  # values at all - only the target's own missingness (ry) trims rows.
  # Checking rank on raw df with complete.cases() checks a DIFFERENT,
  # coincidentally row-starved matrix (listwise deletion across every
  # candidate predictor at once) that can look rank-deficient - or, as
  # confirmed against real data, look FULL rank - for reasons that have
  # nothing to do with whether mice's actual working matrix is singular.
  # This fill is only for that structural rank check below, never used for
  # anything mice itself sees.
  df_filled <- df
  for (col in colnames(df_filled)) {
    x <- df_filled[[col]]
    if (!any(is.na(x))) next
    if (is.numeric(x)) {
      x[is.na(x)] <- mean(x, na.rm = TRUE)
    } else {
      tab <- table(x, useNA = "no")
      mode_val <- if (length(tab) > 0) names(tab)[which.max(tab)] else NA
      x[is.na(x)] <- mode_val
    }
    df_filled[[col]] <- x
  }
  for (target in rownames(pred_matrix)) {
    candidates <- which(pred_matrix[target, ] == 1)
    if (length(candidates) == 0) next
    ranked <- candidates[order(cor_matrix[target, candidates], decreasing = TRUE)]
    kept <- integer(0)
    for (cand in ranked) {
      if (length(kept) >= N_MAX_PREDICTORS) break
      if (length(kept) == 0 || all(cor_matrix[cand, kept] < MAX_PREDICTOR_INTERCORRELATION)) {
        kept <- c(kept, cand)
      }
    }

    # Final guard: a GROUP of predictors can be jointly rank-deficient with
    # no single pair highly correlated - confirmed against real data, this
    # is exactly what happens with Scorecard's "compositional" families
    # that sum to a constant by construction (PCIP*, UGDS_*, IRPS_* -
    # percent of degrees/enrollment/recipients by field or demographic
    # group). Pairwise correlation can't see a sum-to-100% constraint. Drop
    # the least-target-correlated kept predictor, one at a time, until the
    # predictor submatrix's actual numeric rank matches its column count -
    # this catches ANY exact linear dependency regardless of source, not
    # just pairwise correlation, and is the only fully general fix.
    if (length(kept) > 1) {
      kept_names <- colnames(pred_matrix)[kept]
      repeat {
        if (length(kept_names) <= 1) break
        X <- data.matrix(df_filled[, kept_names, drop = FALSE])
        r <- qr(X)$rank
        if (r >= ncol(X)) break
        # kept_names is ranked by correlation with target, descending -
        # drop the weakest (last) predictor first.
        kept_names <- kept_names[-length(kept_names)]
      }
      kept <- match(kept_names, colnames(pred_matrix))
    }

    pred_matrix[target, ] <- 0
    pred_matrix[target, kept] <- 1
  }

  # RIDGE: still worth keeping above mice's default (1e-5) as a backstop
  # even after the mincor fix above - some redundancy among Scorecard
  # variables is real and not fully removable by a single correlation
  # threshold. 1e-4 is the standard first troubleshooting step for
  # singularity in mice; raise further (1e-3) if it recurs.
  imp <- mice::mice(
    df,
    m = N_DATASETS,
    maxit = N_ITERATIONS,
    method = method_vec,
    predictorMatrix = pred_matrix,
    seed = RANDOM_STATE,
    ridge = 1e-3,
    printFlag = TRUE
  )

  completed <- vector("list", N_DATASETS)
  unitids <- rownames(df)
  for (i in seq_len(N_DATASETS)) {
    d <- mice::complete(imp, action = i)
    for (col in numeric_cols) {
      if (!(col %in% colnames(d))) next
      d[[col]] <- d[[col]] * scale_sd[col] + scale_center[col]
    }
    for (var in names(rank_maps)) {
      if (!(var %in% colnames(d))) next
      rank_map <- rank_maps[[var]]
      lo <- min(rank_map)
      hi <- max(rank_map)
      d[[var]] <- pmin(pmax(round(d[[var]]), lo), hi)
    }
    d <- data.frame(UNITID = unitids, d, row.names = NULL, check.names = FALSE)
    completed[[i]] <- d
  }
  list(imp = imp, completed = completed)
}

# -----------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------

build_analysis_frame <- function(raw_csv = RAW_CSV, dictionary_csv = DICTIONARY_CSV) {
  df <- load_raw(raw_csv, dictionary_csv)

  split_result <- split_metadata_columns(df)
  df <- split_result$mice_df
  metadata_df <- split_result$metadata_df
  metadata_cols <- split_result$metadata_cols
  if (length(metadata_cols) > 0) {
    cat(sprintf(
      "Set aside %d per-variable metadata column(s) (matching %s) - excluded from MICE, not deleted.\n",
      length(metadata_cols), paste(METADATA_COLUMN_SUFFIXES, collapse = ",")
    ))
  }

  fixed <- apply_structural_fixes(df)
  df <- fixed$df
  rank_maps <- fixed$rank_maps

  sparsity_report <- report_sparsity(df)

  if (APPLY_COVERAGE_FILTER) {
    cov_result <- select_by_coverage(df, COVERAGE_THRESHOLD)
    df <- cov_result$df
    dropped_coverage <- cov_result$dropped
  } else {
    dropped_coverage <- numeric(0)
  }

  columns <- classify_columns(df)

  degen <- drop_degenerate_columns(df, c(columns$nominal, columns$binary))
  df <- degen$df
  columns$nominal <- setdiff(columns$nominal, degen$dropped)
  columns$binary <- setdiff(columns$binary, degen$dropped)

  nearconst <- drop_near_constant_continuous(df, columns$continuous)
  df <- nearconst$df
  columns$continuous <- setdiff(columns$continuous, nearconst$dropped)

  df <- finalize_types(df, columns)

  list(
    df = df,
    rank_maps = rank_maps,
    columns = columns,
    metadata_df = metadata_df,
    metadata_cols = metadata_cols,
    sparsity_report = sparsity_report,
    dropped_coverage = dropped_coverage,
    dropped_degenerate = degen$dropped,
    dropped_near_constant = nearconst$dropped
  )
}

main <- function() {
  dir.create(OUTPUT_DIR, showWarnings = FALSE)

  result <- build_analysis_frame(RAW_CSV, DICTIONARY_CSV)
  df <- result$df

  cat(sprintf("Columns kept: %d | rows: %d\n", ncol(df), nrow(df)))
  cat(sprintf("Columns dropped for coverage < %.0f%%: %d\n", COVERAGE_THRESHOLD * 100, length(result$dropped_coverage)))
  cat(sprintf(
    "Columns dropped for a category with < %d rows (or near-constant continuous): %d\n",
    MIN_CATEGORY_COUNT, length(result$dropped_degenerate) + length(result$dropped_near_constant)
  ))
  for (kind in names(result$columns)) {
    cat(sprintf("  %s: %d\n", kind, length(result$columns[[kind]])))
  }

  write.csv(result$sparsity_report, file.path(OUTPUT_DIR, "sparsity_report.csv"), row.names = FALSE)
  if (length(result$metadata_cols) > 0) {
    write.csv(
      data.frame(UNITID = rownames(result$metadata_df), result$metadata_df, row.names = NULL, check.names = FALSE),
      file.path(OUTPUT_DIR, "metadata_columns.csv"), row.names = FALSE
    )
  }

  mice_result <- run_mice_r(df, result$columns, result$rank_maps)
  for (i in seq_along(mice_result$completed)) {
    write.csv(mice_result$completed[[i]], file.path(OUTPUT_DIR, sprintf("completed_%d.csv", i - 1)), row.names = FALSE)
  }
  saveRDS(mice_result$imp, file.path(OUTPUT_DIR, "mice_object.rds"))

  cat(sprintf("Wrote %d completed datasets to %s/\n", N_DATASETS, OUTPUT_DIR))
}

if (sys.nframe() == 0) {
  main()
}
