#This calculates:
#- Krippendorff’s α (ordinal) for each C1–C8.
#- Overall ordinal α across all criterion ratings.
#- ICC(2,1) for the 48 total SOP scores — appropriate when the same fixed set of SOPs is evaluated by the same three judges and we care about absolute agreement.
#- ICC(2,3) for the mean of the three judges — useful because our eventual SOP score will aggregate all three judges.
#- It explicitly detects zero-variance judge/criterion combinations such as Gemini C4/C5.
#- No API calls and no modification of existing data.

import csv
import math
import statistics
from collections import Counter
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "FINAL_EVALUATION_RESULTS"
    / "CONSOLIDATED"
    / "evaluation_summary_144.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "FINAL_EVALUATION_RESULTS"
    / "ANALYSIS"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "inter_judge_agreement.csv"
)

JUDGES = [
    "GPT",
    "CLAUDE",
    "GEMINI"
]

CRITERIA = [
    "C1_completeness",
    "C2_correct_sequence",
    "C3_role_clarity",
    "C4_input_output_clarity",
    "C5_exception_handling",
    "C6_constraint_adherence",
    "C7_groundedness",
    "C8_usability",
]

EXPECTED_IDS = [
    f"SOP_{i:03d}"
    for i in range(1, 49)
]


# ============================================================
# 2. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

with INPUT_FILE.open(
    "r",
    newline="",
    encoding="utf-8"
) as f:

    rows = list(
        csv.DictReader(f)
    )


if len(rows) != 144:
    raise ValueError(
        f"Expected 144 rows, found {len(rows)}."
    )


# ============================================================
# 3. BUILD SOP × JUDGE STRUCTURE
# ============================================================

data = {}

for row in rows:

    anonymous_id = row["anonymous_id"]
    judge = row["judge"]

    if anonymous_id not in EXPECTED_IDS:
        raise ValueError(
            f"Unexpected candidate ID: {anonymous_id}"
        )

    if judge not in JUDGES:
        raise ValueError(
            f"Unexpected judge: {judge}"
        )

    if anonymous_id not in data:
        data[anonymous_id] = {}

    if judge in data[anonymous_id]:
        raise ValueError(
            f"Duplicate evaluation: "
            f"{anonymous_id} / {judge}"
        )

    criterion_scores = {}

    for criterion in CRITERIA:

        column = f"{criterion}_score"

        score = int(row[column])

        if score not in [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Invalid score: "
                f"{anonymous_id} / "
                f"{judge} / "
                f"{criterion}"
            )

        criterion_scores[criterion] = score

    total_score = int(
        row["total_score"]
    )

    calculated_total = sum(
        criterion_scores.values()
    )

    if total_score != calculated_total:
        raise ValueError(
            f"Total mismatch: "
            f"{anonymous_id} / {judge}"
        )

    data[anonymous_id][judge] = {
        "total": total_score,
        "criteria": criterion_scores,
    }


# Verify all 48 × 3 evaluations

if set(data.keys()) != set(EXPECTED_IDS):
    raise ValueError(
        "Dataset does not contain exactly "
        "SOP_001 through SOP_048."
    )

for anonymous_id in EXPECTED_IDS:

    if set(data[anonymous_id].keys()) != set(JUDGES):

        raise ValueError(
            f"{anonymous_id} does not have "
            f"exactly all three judges."
        )


print("=" * 74)
print("STEP 2 — INTER-JUDGE AGREEMENT")
print("=" * 74)

print()
print("Dataset verified:")
print("  SOPs:                 48")
print("  Judges per SOP:        3")
print("  Evaluations:          144")
print("  Criterion ratings:  1,152")


# ============================================================
# 4. CHECK JUDGE VARIANCE BY CRITERION
# ============================================================

print()
print("=" * 74)
print("A. ZERO-VARIANCE CHECK")
print("=" * 74)

zero_variance_cases = []

for criterion in CRITERIA:

    for judge in JUDGES:

        scores = [
            data[anonymous_id][judge][
                "criteria"
            ][criterion]
            for anonymous_id in EXPECTED_IDS
        ]

        unique_scores = sorted(
            set(scores)
        )

        if len(unique_scores) == 1:

            zero_variance_cases.append(
                (
                    judge,
                    criterion,
                    unique_scores[0]
                )
            )

            print(
                f"{judge} / {criterion}: "
                f"ZERO VARIANCE "
                f"(all scores = {unique_scores[0]})"
            )

if not zero_variance_cases:
    print(
        "No zero-variance judge/criterion "
        "combinations detected."
    )


# ============================================================
# 5. KRIPPENDORFF'S ALPHA — ORDINAL
# ============================================================
#
# Ordinal disagreement is calculated from the
# observed marginal category frequencies.
#
# For ordered categories c and k:
#
# delta(c,k) =
# [sum of marginal frequencies between c and k
#  minus half the endpoint frequencies]^2
#
# The common normalization factor cancels in
# alpha = 1 - Do/De.
#
# This implementation uses complete data:
# 48 units × 3 judges.
# ============================================================

def ordinal_distance_matrix(ratings):

    categories = sorted(
        set(ratings)
    )

    frequencies = Counter(
        ratings
    )

    distances = {}

    for c in categories:

        for k in categories:

            if c == k:

                distances[(c, k)] = 0.0
                continue

            low = min(c, k)
            high = max(c, k)

            cumulative = 0.0

            for category in categories:

                if low <= category <= high:
                    cumulative += frequencies[
                        category
                    ]

            cumulative -= (
                frequencies[low] / 2.0
            )

            cumulative -= (
                frequencies[high] / 2.0
            )

            distances[(c, k)] = (
                cumulative ** 2
            )

    return distances


def krippendorff_alpha_ordinal(
    units
):

    # units =
    # [
    #   [judge1, judge2, judge3],
    #   ...
    # ]

    all_ratings = [
        rating
        for unit in units
        for rating in unit
    ]

    if len(set(all_ratings)) <= 1:
        return float("nan")

    distances = ordinal_distance_matrix(
        all_ratings
    )

    # --------------------------------------------------------
    # Observed disagreement
    # --------------------------------------------------------

    observed_sum = 0.0
    observed_pairs = 0

    for unit in units:

        n = len(unit)

        if n < 2:
            continue

        for i in range(n):

            for j in range(n):

                if i == j:
                    continue

                observed_sum += distances[
                    (unit[i], unit[j])
                ]

                observed_pairs += 1

    if observed_pairs == 0:
        return float("nan")

    Do = (
        observed_sum
        / observed_pairs
    )

    # --------------------------------------------------------
    # Expected disagreement
    #
    # Draw two distinct ratings from the full
    # pooled marginal distribution.
    # --------------------------------------------------------

    expected_sum = 0.0
    expected_pairs = 0

    N = len(all_ratings)

    for i in range(N):

        for j in range(N):

            if i == j:
                continue

            expected_sum += distances[
                (
                    all_ratings[i],
                    all_ratings[j]
                )
            ]

            expected_pairs += 1

    if expected_pairs == 0:
        return float("nan")

    De = (
        expected_sum
        / expected_pairs
    )

    if De == 0:
        return float("nan")

    alpha = 1.0 - (Do / De)

    return alpha


# ============================================================
# 6. ALPHA FOR EACH CRITERION
# ============================================================

print()
print("=" * 74)
print("B. KRIPPENDORFF'S ALPHA — ORDINAL")
print("=" * 74)

print()
print(
    "Criterion-level agreement across "
    "GPT, Claude, and Gemini:"
)
print()

alpha_results = {}


for criterion in CRITERIA:

    units = []

    for anonymous_id in EXPECTED_IDS:

        unit = [
            data[anonymous_id][judge][
                "criteria"
            ][criterion]
            for judge in JUDGES
        ]

        units.append(unit)

    alpha = krippendorff_alpha_ordinal(
        units
    )

    alpha_results[criterion] = alpha

    if math.isnan(alpha):

        alpha_text = "undefined"

    else:

        alpha_text = f"{alpha:.4f}"

    print(
        f"{criterion}: {alpha_text}"
    )


# ============================================================
# 7. OVERALL CRITERION-LEVEL ALPHA
# ============================================================
#
# Each SOP × criterion pair is treated as one
# evaluation unit rated by the three judges.
#
# 48 SOPs × 8 criteria = 384 units.
# ============================================================

overall_units = []

for anonymous_id in EXPECTED_IDS:

    for criterion in CRITERIA:

        unit = [
            data[anonymous_id][judge][
                "criteria"
            ][criterion]
            for judge in JUDGES
        ]

        overall_units.append(
            unit
        )


overall_alpha = (
    krippendorff_alpha_ordinal(
        overall_units
    )
)


print()
print(
    "Overall ordinal alpha "
    "(384 SOP × criterion units):"
)

if math.isnan(overall_alpha):

    print("undefined")

else:

    print(
        f"{overall_alpha:.4f}"
    )


# ============================================================
# 8. ICC(2,1) AND ICC(2,3) FOR TOTAL SCORES
# ============================================================
#
# Two-way random-effects, absolute agreement.
#
# n = 48 targets (SOPs)
# k = 3 judges
#
# ICC(2,1):
#
# (MSR - MSE) /
# [MSR + (k-1)MSE +
#  k(MSC-MSE)/n]
#
# ICC(2,k):
#
# (MSR - MSE) /
# [MSR + (MSC-MSE)/n]
#
# Here k = 3.
# ============================================================

matrix = []

for anonymous_id in EXPECTED_IDS:

    matrix.append(
        [
            data[anonymous_id][judge][
                "total"
            ]
            for judge in JUDGES
        ]
    )


n = len(matrix)
k = len(JUDGES)


grand_mean = statistics.mean(
    value
    for row in matrix
    for value in row
)


row_means = [
    statistics.mean(row)
    for row in matrix
]


column_means = []

for j in range(k):

    column_values = [
        matrix[i][j]
        for i in range(n)
    ]

    column_means.append(
        statistics.mean(
            column_values
        )
    )


# ------------------------------------------------------------
# Sum of squares
# ------------------------------------------------------------

SS_rows = (
    k
    * sum(
        (row_mean - grand_mean) ** 2
        for row_mean in row_means
    )
)


SS_columns = (
    n
    * sum(
        (column_mean - grand_mean) ** 2
        for column_mean in column_means
    )
)


SS_total = sum(
    (
        matrix[i][j]
        - grand_mean
    ) ** 2
    for i in range(n)
    for j in range(k)
)


SS_error = (
    SS_total
    - SS_rows
    - SS_columns
)


# ------------------------------------------------------------
# Degrees of freedom
# ------------------------------------------------------------

df_rows = n - 1
df_columns = k - 1
df_error = (
    (n - 1)
    * (k - 1)
)


# ------------------------------------------------------------
# Mean squares
# ------------------------------------------------------------

MSR = (
    SS_rows
    / df_rows
)

MSC = (
    SS_columns
    / df_columns
)

MSE = (
    SS_error
    / df_error
)


# ------------------------------------------------------------
# ICC(2,1)
# ------------------------------------------------------------

icc_2_1_denominator = (
    MSR
    + (k - 1) * MSE
    + (
        k
        * (MSC - MSE)
        / n
    )
)

ICC_2_1 = (
    (MSR - MSE)
    / icc_2_1_denominator
)


# ------------------------------------------------------------
# ICC(2,k)
# ------------------------------------------------------------

icc_2_k_denominator = (
    MSR
    + (
        (MSC - MSE)
        / n
    )
)

ICC_2_k = (
    (MSR - MSE)
    / icc_2_k_denominator
)


print()
print("=" * 74)
print("C. ICC — TOTAL SOP SCORES")
print("=" * 74)

print()
print(
    "Two-way random-effects, "
    "absolute-agreement ICC"
)

print()

print(
    f"ICC(2,1) — single judge: "
    f"{ICC_2_1:.4f}"
)

print(
    f"ICC(2,3) — mean of 3 judges: "
    f"{ICC_2_k:.4f}"
)

print()
print("ANOVA components:")
print(f"  MSR (SOPs):   {MSR:.6f}")
print(f"  MSC (judges): {MSC:.6f}")
print(f"  MSE (error):  {MSE:.6f}")


# ============================================================
# 9. PAIRWISE TOTAL-SCORE CORRELATIONS
# ============================================================
#
# These are descriptive only.
# Correlation measures association, not absolute
# agreement.
# ============================================================

def pearson_correlation(x, y):

    if len(x) != len(y):
        raise ValueError(
            "Vectors have different lengths."
        )

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    numerator = sum(
        (a - mean_x)
        * (b - mean_y)
        for a, b in zip(x, y)
    )

    denominator_x = math.sqrt(
        sum(
            (a - mean_x) ** 2
            for a in x
        )
    )

    denominator_y = math.sqrt(
        sum(
            (b - mean_y) ** 2
            for b in y
        )
    )

    denominator = (
        denominator_x
        * denominator_y
    )

    if denominator == 0:
        return float("nan")

    return (
        numerator
        / denominator
    )


print()
print("=" * 74)
print("D. PAIRWISE TOTAL-SCORE CORRELATIONS")
print("=" * 74)

print()
print(
    "Pearson correlations are reported only "
    "as supplementary descriptive statistics."
)
print()


pairwise_correlations = []


for i in range(len(JUDGES)):

    for j in range(
        i + 1,
        len(JUDGES)
    ):

        judge_a = JUDGES[i]
        judge_b = JUDGES[j]

        scores_a = [
            data[anonymous_id][judge_a][
                "total"
            ]
            for anonymous_id in EXPECTED_IDS
        ]

        scores_b = [
            data[anonymous_id][judge_b][
                "total"
            ]
            for anonymous_id in EXPECTED_IDS
        ]

        r = pearson_correlation(
            scores_a,
            scores_b
        )

        pairwise_correlations.append(
            (
                judge_a,
                judge_b,
                r
            )
        )

        if math.isnan(r):
            r_text = "undefined"
        else:
            r_text = f"{r:.4f}"

        print(
            f"{judge_a} vs {judge_b}: "
            f"r = {r_text}"
        )


# ============================================================
# 10. SAVE AGREEMENT RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


output_rows = []


for criterion in CRITERIA:

    alpha = alpha_results[
        criterion
    ]

    output_rows.append({
        "analysis":
            "Krippendorff_alpha_ordinal",

        "dimension":
            criterion,

        "value":
            (
                ""
                if math.isnan(alpha)
                else round(alpha, 6)
            )
    })


output_rows.append({
    "analysis":
        "Krippendorff_alpha_ordinal",

    "dimension":
        "overall_384_units",

    "value":
        (
            ""
            if math.isnan(overall_alpha)
            else round(
                overall_alpha,
                6
            )
        )
})


output_rows.append({
    "analysis":
        "ICC_2_1_absolute_agreement",

    "dimension":
        "total_score",

    "value":
        round(
            ICC_2_1,
            6
        )
})


output_rows.append({
    "analysis":
        "ICC_2_3_absolute_agreement",

    "dimension":
        "total_score_mean_3_judges",

    "value":
        round(
            ICC_2_k,
            6
        )
})


for judge_a, judge_b, r in pairwise_correlations:

    output_rows.append({
        "analysis":
            "Pearson_total_score",

        "dimension":
            f"{judge_a}_vs_{judge_b}",

        "value":
            (
                ""
                if math.isnan(r)
                else round(
                    r,
                    6
                )
            )
    })


with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "analysis",
            "dimension",
            "value",
        ]
    )

    writer.writeheader()
    writer.writerows(
        output_rows
    )


# ============================================================
# 11. FINAL REPORT
# ============================================================

print()
print("=" * 74)
print("STEP 2 COMPLETE")
print("=" * 74)

print()
print("Created:")
print(OUTPUT_FILE)

print()
print(
    "Zero-variance cases detected: "
    f"{len(zero_variance_cases)}"
)

print()
print("No API requests were made.")
print("No evaluation results were modified.")

print()
print(
    "Send the complete console output "
    "before proceeding to model comparison."
)

print("=" * 74)

#What the results mean
#The three judges have limited absolute agreement overall.
#Your criterion-level Krippendorff’s α values range from −0.108 to 0.539. Agreement is strongest for C6 Constraint Adherence (α = 0.539) and C1 Completeness (α = 0.507). Several criteria are around zero or negative, particularly C7 Groundedness (α = −0.108). The overall α is only 0.195.
#The total-score results tell a similar story:
#- ICC(2,1) = 0.243 → a single judge has low absolute agreement with the other judges.
#- ICC(2,3) = 0.490 → averaging all three improves reliability, but agreement is still not strong.
#- Yet pairwise correlations are around 0.50–0.56. So the judges have some tendency to identify the same SOPs as relatively stronger/weaker, while assigning substantially different absolute scores.
#The huge MSC = 159.90 compared with MSE = 2.46 is particularly revealing: there is a strong systematic judge severity effect. That matches Step 1 perfectly: GPT averaged 36.02, Claude 38.00, and Gemini 39.67.
#This is not a failed experiment
#Actually, this gives us an important methodological finding. It demonstrates exactly why relying on one LLM judge would have been risky.
#We should not simply average the three raw judge totals and treat that average as unquestionable ground truth. Gemini's ceiling effect would systematically push scores upward, while GPT uses the scale much more strictly.
#For the thesis, we can later state that the multi-judge design exposed substantial judge-specific calibration differences and therefore individual judge scores were retained rather than assuming interchangeability.
#Before model comparison
#There is one thing I want to add before aggregating anything: pairwise weighted Cohen's κ for C1–C8.
#Why? Krippendorff's α tells us overall three-judge agreement, but it doesn't tell us whether the problem is mainly:
#GPT ↔ Claude, GPT ↔ Gemini, or Claude ↔ Gemini.
#That becomes particularly useful given Gemini's ceiling effect and will strengthen the methodological analysis.
#After that,
# we can make a defensible decision about how to aggregate the judges and then finally compare the four generating models.