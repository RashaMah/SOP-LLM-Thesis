from pathlib import Path
import pandas as pd
import numpy as np
import krippendorff


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "FINAL_EVALUATION_RESULTS"
    / "CONSOLIDATED"
    / "criterion_ratings_1152.csv"
)


# ============================================================
# 2. REFERENCE VALUES FROM ORIGINAL ANALYSIS
# ============================================================

ORIGINAL_ALPHA = {
    "C1": 0.5069,
    "C2": 0.3102,
    "C3": -0.0293,
    "C4": -0.0515,
    "C5": -0.0366,
    "C6": 0.5387,
    "C7": -0.1082,
    "C8": 0.0035,
    "OVERALL": 0.1954,
}


# ============================================================
# 3. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 78)
print("INDEPENDENT KRIPPENDORFF ALPHA VALIDATION")
print("=" * 78)

print(f"\nLoaded criterion ratings: {len(df)}")

if len(df) != 1152:
    raise ValueError(
        f"Expected 1152 criterion ratings, found {len(df)}."
    )

print("\nColumns found:")
print(list(df.columns))


# ============================================================
# 4. DETECT REQUIRED COLUMNS
# ============================================================

def find_column(options):
    for option in options:
        if option in df.columns:
            return option

    raise ValueError(
        "Could not find any of these columns: "
        + ", ".join(options)
    )


SOP_COL = find_column(
    ["anonymous_id", "sop_id"]
)

JUDGE_COL = find_column(
    ["judge", "judge_model"]
)

CRITERION_COL = find_column(
    ["criterion", "criterion_id"]
)

SCORE_COL = find_column(
    ["score", "criterion_score"]
)

print("\nDetected columns:")
print(f"  SOP:       {SOP_COL}")
print(f"  Judge:     {JUDGE_COL}")
print(f"  Criterion: {CRITERION_COL}")
print(f"  Score:     {SCORE_COL}")


# ============================================================
# 5. NORMALIZE CRITERION LABELS
# ============================================================

def normalize_criterion(value):
    value = str(value).strip().upper()

    for criterion in [
        "C1", "C2", "C3", "C4",
        "C5", "C6", "C7", "C8"
    ]:
        if value.startswith(criterion):
            return criterion

    return value


df["criterion_normalized"] = (
    df[CRITERION_COL]
    .apply(normalize_criterion)
)


# ============================================================
# 6. BASIC DESIGN CHECK
# ============================================================

criteria = [
    "C1", "C2", "C3", "C4",
    "C5", "C6", "C7", "C8"
]

judges = sorted(
    df[JUDGE_COL]
    .astype(str)
    .unique()
)

sops = sorted(
    df[SOP_COL]
    .astype(str)
    .unique()
)

print("\nDesign:")
print(f"  SOPs:       {len(sops)}")
print(f"  Judges:     {len(judges)}")
print(f"  Criteria:   {len(criteria)}")

if len(sops) != 48:
    raise ValueError(
        f"Expected 48 SOPs, found {len(sops)}."
    )

if len(judges) != 3:
    raise ValueError(
        f"Expected 3 judges, found {len(judges)}."
    )

for criterion in criteria:

    count = (
        df["criterion_normalized"]
        == criterion
    ).sum()

    if count != 144:
        raise ValueError(
            f"{criterion}: expected 144 ratings, "
            f"found {count}."
        )

print("\nDesign validation passed.")


# ============================================================
# 7. FUNCTION TO BUILD RELIABILITY MATRIX
# ============================================================
#
# krippendorff.alpha expects:
#
# rows    = raters
# columns = units
#
# For each criterion:
#   3 judges × 48 SOPs
#
# ============================================================

def build_matrix(data, unit_column):

    pivot = data.pivot(
        index=JUDGE_COL,
        columns=unit_column,
        values=SCORE_COL,
    )

    if pivot.isnull().any().any():
        raise ValueError(
            "Missing ratings detected in reliability matrix."
        )

    return pivot.astype(float).to_numpy()


# ============================================================
# 8. CRITERION-SPECIFIC ORDINAL ALPHA
# ============================================================

print()
print("=" * 78)
print("A. CRITERION-SPECIFIC ORDINAL ALPHA")
print("=" * 78)

validated = {}

print()
print(
    f"{'Criterion':<12}"
    f"{'Original':>12}"
    f"{'Package':>12}"
    f"{'Difference':>14}"
)

print("-" * 50)

for criterion in criteria:

    subset = df[
        df["criterion_normalized"]
        == criterion
    ].copy()

    matrix = build_matrix(
        subset,
        SOP_COL
    )

    alpha = krippendorff.alpha(
        reliability_data=matrix,
        level_of_measurement="ordinal",
    )

    alpha = float(alpha)

    validated[criterion] = alpha

    original = ORIGINAL_ALPHA[
        criterion
    ]

    difference = (
        alpha - original
    )

    print(
        f"{criterion:<12}"
        f"{original:>12.6f}"
        f"{alpha:>12.6f}"
        f"{difference:>+14.6f}"
    )


# ============================================================
# 9. OVERALL ORDINAL ALPHA
# ============================================================
#
# Unit = SOP × criterion.
#
# This reproduces the overall design:
# 48 SOPs × 8 criteria = 384 units,
# each rated by 3 judges.
# ============================================================

overall_df = df.copy()

overall_df["unit_id"] = (
    overall_df[SOP_COL].astype(str)
    + "__"
    + overall_df["criterion_normalized"]
)

overall_matrix = build_matrix(
    overall_df,
    "unit_id"
)

overall_alpha = krippendorff.alpha(
    reliability_data=overall_matrix,
    level_of_measurement="ordinal",
)

overall_alpha = float(
    overall_alpha
)

validated["OVERALL"] = overall_alpha


print()
print("=" * 78)
print("B. OVERALL ORDINAL ALPHA")
print("=" * 78)

original_overall = (
    ORIGINAL_ALPHA["OVERALL"]
)

overall_difference = (
    overall_alpha
    - original_overall
)

print()
print(
    f"Original analysis:  "
    f"{original_overall:.6f}"
)

print(
    f"krippendorff pkg:   "
    f"{overall_alpha:.6f}"
)

print(
    f"Difference:         "
    f"{overall_difference:+.6f}"
)


# ============================================================
# 10. VALIDATION CHECK
# ============================================================
#
# Original values were saved rounded to four decimals.
# Therefore a difference below 0.0001 is considered an
# expected rounding-level match.
# ============================================================

TOLERANCE = 0.0001

all_match = True

print()
print("=" * 78)
print("C. VALIDATION STATUS")
print("=" * 78)
print()

for key in [
    "C1", "C2", "C3", "C4",
    "C5", "C6", "C7", "C8",
    "OVERALL",
]:

    difference = abs(
        validated[key]
        - ORIGINAL_ALPHA[key]
    )

    match = (
        difference <= TOLERANCE
    )

    if not match:
        all_match = False

    print(
        f"{key:<8} "
        f"{'MATCH' if match else 'CHECK'} "
        f"(absolute difference = "
        f"{difference:.6f})"
    )


# ============================================================
# 11. FINAL RESULT
# ============================================================

print()
print("=" * 78)

if all_match:

    print(
        "VALIDATION PASSED"
    )

    print()
    print(
        "The independent krippendorff package "
        "reproduces the original ordinal-alpha "
        "results within rounding tolerance."
    )

else:

    print(
        "VALIDATION REQUIRES REVIEW"
    )

    print()
    print(
        "At least one independently calculated "
        "alpha differs from the original result "
        "by more than the rounding tolerance."
    )

    print(
        "Do NOT change the thesis values until "
        "the discrepancy has been investigated."
    )

print()
print("No source data were modified.")
print("=" * 78)