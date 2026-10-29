import csv
import math
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
    / "pairwise_weighted_kappa.csv"
)

JUDGES = [
    "GPT",
    "CLAUDE",
    "GEMINI"
]

JUDGE_PAIRS = [
    ("GPT", "CLAUDE"),
    ("GPT", "GEMINI"),
    ("CLAUDE", "GEMINI"),
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

    rows = list(csv.DictReader(f))


if len(rows) != 144:
    raise ValueError(
        f"Expected 144 rows, found {len(rows)}."
    )


# ============================================================
# 3. BUILD SOP × JUDGE DATA
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

        score = int(
            row[f"{criterion}_score"]
        )

        if score not in [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Invalid score: "
                f"{anonymous_id} / "
                f"{judge} / "
                f"{criterion}"
            )

        criterion_scores[criterion] = score

    data[anonymous_id][judge] = (
        criterion_scores
    )


# Verify completeness

if set(data.keys()) != set(EXPECTED_IDS):
    raise ValueError(
        "Dataset does not contain exactly "
        "SOP_001 through SOP_048."
    )

for anonymous_id in EXPECTED_IDS:

    if set(data[anonymous_id].keys()) != set(JUDGES):
        raise ValueError(
            f"{anonymous_id}: incomplete judge data."
        )


# ============================================================
# 4. QUADRATIC WEIGHTED COHEN'S KAPPA
# ============================================================
#
# Ratings are ordinal: 1–5.
#
# Quadratic disagreement weight:
#
#     w(i,j) = ((i-j)/(k-1))²
#
# Kappa:
#
#     κ = 1 - observed_disagreement /
#             expected_disagreement
#
# A value of 1 = perfect agreement.
# 0 = agreement expected by chance.
# Negative = worse than chance expectation.
#
# ============================================================

def quadratic_weighted_kappa(
    ratings_a,
    ratings_b
):

    if len(ratings_a) != len(ratings_b):
        raise ValueError(
            "Rating vectors have different lengths."
        )

    n = len(ratings_a)

    if n == 0:
        return float("nan")

    categories = [1, 2, 3, 4, 5]
    k = len(categories)

    # --------------------------------------------------------
    # Observed confusion matrix
    # --------------------------------------------------------

    observed = [
        [0 for _ in categories]
        for _ in categories
    ]

    for a, b in zip(
        ratings_a,
        ratings_b
    ):

        observed[a - 1][b - 1] += 1


    # --------------------------------------------------------
    # Marginal frequencies
    # --------------------------------------------------------

    hist_a = [0] * k
    hist_b = [0] * k

    for value in ratings_a:
        hist_a[value - 1] += 1

    for value in ratings_b:
        hist_b[value - 1] += 1


    # --------------------------------------------------------
    # Observed weighted disagreement
    # --------------------------------------------------------

    observed_disagreement = 0.0

    for i in range(k):

        for j in range(k):

            weight = (
                ((i - j) / (k - 1)) ** 2
            )

            observed_probability = (
                observed[i][j] / n
            )

            observed_disagreement += (
                weight
                * observed_probability
            )


    # --------------------------------------------------------
    # Expected weighted disagreement
    # --------------------------------------------------------

    expected_disagreement = 0.0

    for i in range(k):

        for j in range(k):

            weight = (
                ((i - j) / (k - 1)) ** 2
            )

            expected_probability = (
                (hist_a[i] / n)
                * (hist_b[j] / n)
            )

            expected_disagreement += (
                weight
                * expected_probability
            )


    if expected_disagreement == 0:
        return float("nan")


    kappa = (
        1
        - (
            observed_disagreement
            / expected_disagreement
        )
    )

    return kappa


# ============================================================
# 5. CALCULATE PAIRWISE KAPPA
# ============================================================

print("=" * 76)
print("STEP 2B — PAIRWISE QUADRATIC WEIGHTED COHEN'S KAPPA")
print("=" * 76)

print()
print("Dataset verified:")
print("  SOPs:              48")
print("  Judges:             3")
print("  Criteria:           8")
print("  Judge pairs:        3")

results = []


for judge_a, judge_b in JUDGE_PAIRS:

    print()
    print("=" * 76)
    print(f"{judge_a} vs {judge_b}")
    print("=" * 76)

    pair_values = []

    for criterion in CRITERIA:

        ratings_a = [
            data[anonymous_id][
                judge_a
            ][criterion]
            for anonymous_id
            in EXPECTED_IDS
        ]

        ratings_b = [
            data[anonymous_id][
                judge_b
            ][criterion]
            for anonymous_id
            in EXPECTED_IDS
        ]

        kappa = (
            quadratic_weighted_kappa(
                ratings_a,
                ratings_b
            )
        )

        pair_values.append(
            kappa
        )

        if math.isnan(kappa):
            kappa_text = "undefined"
        else:
            kappa_text = f"{kappa:.4f}"

        print(
            f"{criterion}: "
            f"{kappa_text}"
        )

        results.append({
            "judge_a": judge_a,
            "judge_b": judge_b,
            "criterion": criterion,
            "weighted_kappa": (
                ""
                if math.isnan(kappa)
                else round(kappa, 6)
            ),
        })


# ============================================================
# 6. OVERALL PAIRWISE KAPPA
# ============================================================
#
# Here all 48 × 8 = 384 criterion ratings
# are pooled for each judge pair.
#
# This is supplementary only because it mixes
# different rubric dimensions.
# ============================================================

print()
print("=" * 76)
print("OVERALL PAIRWISE KAPPA — ALL 384 RATINGS")
print("=" * 76)


for judge_a, judge_b in JUDGE_PAIRS:

    ratings_a = []
    ratings_b = []

    for anonymous_id in EXPECTED_IDS:

        for criterion in CRITERIA:

            ratings_a.append(
                data[anonymous_id][
                    judge_a
                ][criterion]
            )

            ratings_b.append(
                data[anonymous_id][
                    judge_b
                ][criterion]
            )

    kappa = (
        quadratic_weighted_kappa(
            ratings_a,
            ratings_b
        )
    )

    if math.isnan(kappa):
        kappa_text = "undefined"
    else:
        kappa_text = f"{kappa:.4f}"

    print(
        f"{judge_a} vs {judge_b}: "
        f"{kappa_text}"
    )

    results.append({
        "judge_a": judge_a,
        "judge_b": judge_b,
        "criterion": "OVERALL_384_RATINGS",
        "weighted_kappa": (
            ""
            if math.isnan(kappa)
            else round(kappa, 6)
        ),
    })


# ============================================================
# 7. EXACT AGREEMENT PERCENTAGES
# ============================================================

print()
print("=" * 76)
print("EXACT AGREEMENT — SUPPLEMENTARY")
print("=" * 76)


for judge_a, judge_b in JUDGE_PAIRS:

    agreements = 0
    comparisons = 0

    for anonymous_id in EXPECTED_IDS:

        for criterion in CRITERIA:

            score_a = (
                data[anonymous_id][
                    judge_a
                ][criterion]
            )

            score_b = (
                data[anonymous_id][
                    judge_b
                ][criterion]
            )

            comparisons += 1

            if score_a == score_b:
                agreements += 1

    percent = (
        agreements / comparisons
    ) * 100

    print(
        f"{judge_a} vs {judge_b}: "
        f"{agreements}/{comparisons} "
        f"({percent:.1f}%)"
    )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "judge_a",
            "judge_b",
            "criterion",
            "weighted_kappa",
        ]
    )

    writer.writeheader()
    writer.writerows(
        results
    )


# ============================================================
# 9. FINAL REPORT
# ============================================================

print()
print("=" * 76)
print("STEP 2B COMPLETE")
print("=" * 76)

print()
print("Created:")
print(OUTPUT_FILE)

print()
print("No API requests were made.")
print("No evaluation results were modified.")

print()
print(
    "Send the complete console output "
    "before we aggregate judges and "
    "compare the four generator models."
)

print("=" * 76)

#GPT–Claude agree the most overall (weighted κ = 0.438). GPT–Gemini agreement is substantially lower (0.128), while Claude–Gemini is 0.242. Interestingly, Claude–Gemini have 79.4% exact agreement, but their κ remains only 0.242. That apparent contradiction is explained largely by the ceiling effect: both frequently give 5s, so raw percentage agreement can look high even when chance-corrected agreement is limited.
#The criterion results also make sense alongside the earlier analysis. C6 Constraint Adherence is the most consistently judged dimension across pairs (κ = 0.508–0.707). Conversely, several dimensions show weak agreement. Gemini's C4 and C5 κ values of 0 are especially unsurprising because Gemini assigned 5 to every SOP on both criteria—it contributes no discrimination on those dimensions.