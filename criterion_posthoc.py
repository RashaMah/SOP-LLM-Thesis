import csv
import itertools
from pathlib import Path

from scipy.stats import wilcoxon


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "FINAL_EVALUATION_RESULTS"
    / "ANALYSIS"
    / "consensus_sop_scores_48.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "FINAL_EVALUATION_RESULTS"
    / "ANALYSIS"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "criterion_posthoc_comparisons.csv"
)

GENERATORS = [
    "QWEN",
    "LLAMA",
    "GPT",
    "CLAUDE",
]

# Only criteria that survived the Holm-corrected
# omnibus Friedman stage in Step 6.
SIGNIFICANT_CRITERIA = [
    "C1_completeness",
    "C3_role_clarity",
    "C5_exception_handling",
    "C6_constraint_adherence",
    "C7_groundedness",
    "C8_usability",
]

EXPECTED_SCENARIOS = [
    "GR-A",
    "GR-B",
    "RL-A",
    "RL-B",
    "IR-A",
    "IR-B",
    "OP-A",
    "OP-B",
    "ST-A",
    "ST-B",
    "PD-A",
    "PD-B",
]

ALPHA = 0.05


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

if len(rows) != 48:
    raise ValueError(
        f"Expected 48 rows, found {len(rows)}."
    )


# ============================================================
# 3. BUILD MATCHED DATA STRUCTURE
# ============================================================

data = {}

for row in rows:

    scenario = row["scenario_id"]
    generator = row["generator"]

    if scenario not in EXPECTED_SCENARIOS:
        raise ValueError(
            f"Unexpected scenario: {scenario}"
        )

    if generator not in GENERATORS:
        raise ValueError(
            f"Unexpected generator: {generator}"
        )

    key = (scenario, generator)

    if key in data:
        raise ValueError(
            f"Duplicate: {scenario} / {generator}"
        )

    data[key] = {}

    for criterion in SIGNIFICANT_CRITERIA:

        column = f"{criterion}_mean"

        score = float(row[column])

        if not 1 <= score <= 5:
            raise ValueError(
                f"Invalid score: "
                f"{scenario} / "
                f"{generator} / "
                f"{criterion}"
            )

        data[key][criterion] = score


# Verify complete design

for scenario in EXPECTED_SCENARIOS:
    for generator in GENERATORS:

        if (scenario, generator) not in data:
            raise ValueError(
                f"Missing: {scenario} / {generator}"
            )


print("=" * 78)
print("STEP 6B — CRITERION-SPECIFIC POST-HOC COMPARISONS")
print("=" * 78)

print()
print("Design verified:")
print("  Significant criteria:     6")
print("  Generators:                4")
print("  Matched scenarios:        12")
print("  Pairwise tests/criterion:  6")
print("  Total post-hoc tests:     36")


# ============================================================
# 4. RANK-BISERIAL EFFECT SIZE
# ============================================================

def rank_biserial(x, y):

    differences = [
        a - b
        for a, b in zip(x, y)
        if abs(a - b) > 1e-12
    ]

    if not differences:
        return 0.0

    absolute_values = [
        abs(d)
        for d in differences
    ]

    sorted_indices = sorted(
        range(len(absolute_values)),
        key=lambda i: absolute_values[i]
    )

    ranks = [0.0] * len(absolute_values)

    i = 0

    while i < len(sorted_indices):

        j = i

        while (
            j + 1 < len(sorted_indices)
            and abs(
                absolute_values[sorted_indices[j + 1]]
                - absolute_values[sorted_indices[i]]
            ) < 1e-12
        ):
            j += 1

        average_rank = (
            (i + 1) + (j + 1)
        ) / 2.0

        for position in range(i, j + 1):
            ranks[
                sorted_indices[position]
            ] = average_rank

        i = j + 1

    w_plus = sum(
        rank
        for difference, rank
        in zip(differences, ranks)
        if difference > 0
    )

    w_minus = sum(
        rank
        for difference, rank
        in zip(differences, ranks)
        if difference < 0
    )

    total = w_plus + w_minus

    if total == 0:
        return 0.0

    return (
        w_plus - w_minus
    ) / total


# ============================================================
# 5. PAIRWISE TESTS FOR EACH SIGNIFICANT CRITERION
# ============================================================

all_results = []


for criterion in SIGNIFICANT_CRITERIA:

    print()
    print("=" * 78)
    print(criterion)
    print("=" * 78)

    criterion_results = []

    for generator_a, generator_b in itertools.combinations(
        GENERATORS,
        2
    ):

        scores_a = [
            data[
                (scenario, generator_a)
            ][criterion]
            for scenario in EXPECTED_SCENARIOS
        ]

        scores_b = [
            data[
                (scenario, generator_b)
            ][criterion]
            for scenario in EXPECTED_SCENARIOS
        ]

        differences = [
            a - b
            for a, b in zip(
                scores_a,
                scores_b
            )
        ]

        nonzero_count = sum(
            abs(d) > 1e-12
            for d in differences
        )

        mean_difference = (
            sum(differences)
            / len(differences)
        )

        if nonzero_count == 0:

            statistic = 0.0
            raw_p = 1.0

        else:

            result = wilcoxon(
                scores_a,
                scores_b,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",
            )

            statistic = float(
                result.statistic
            )

            raw_p = float(
                result.pvalue
            )

        effect = rank_biserial(
            scores_a,
            scores_b
        )

        criterion_results.append({
            "criterion":
                criterion,

            "generator_a":
                generator_a,

            "generator_b":
                generator_b,

            "n_nonzero":
                nonzero_count,

            "mean_difference_A_minus_B":
                mean_difference,

            "wilcoxon_statistic":
                statistic,

            "raw_p":
                raw_p,

            "rank_biserial":
                effect,
        })


    # ========================================================
    # 6. HOLM CORRECTION WITHIN THIS CRITERION
    # ========================================================

    ordered = sorted(
        range(len(criterion_results)),
        key=lambda i:
            criterion_results[i]["raw_p"]
    )

    m = len(criterion_results)
    previous_adjusted = 0.0

    for position, index in enumerate(
        ordered
    ):

        adjusted = (
            criterion_results[index]["raw_p"]
            * (m - position)
        )

        adjusted = max(
            adjusted,
            previous_adjusted
        )

        adjusted = min(
            adjusted,
            1.0
        )

        criterion_results[
            index
        ]["holm_p"] = adjusted

        criterion_results[
            index
        ]["significant_holm_0.05"] = (
            adjusted < ALPHA
        )

        previous_adjusted = adjusted


    # ========================================================
    # 7. PRINT THIS CRITERION
    # ========================================================

    for result in criterion_results:

        print()
        print(
            f"{result['generator_a']} vs "
            f"{result['generator_b']}"
        )

        print("-" * 45)

        print(
            f"Non-zero pairs:      "
            f"{result['n_nonzero']}"
        )

        print(
            f"Mean difference:     "
            f"{result['mean_difference_A_minus_B']:+.4f}"
        )

        print(
            f"Wilcoxon statistic:  "
            f"{result['wilcoxon_statistic']:.4f}"
        )

        print(
            f"Raw p:               "
            f"{result['raw_p']:.8f}"
        )

        print(
            f"Holm-adjusted p:     "
            f"{result['holm_p']:.8f}"
        )

        print(
            f"Rank-biserial r:     "
            f"{result['rank_biserial']:+.4f}"
        )

        print(
            f"Significant (Holm):  "
            f"{result['significant_holm_0.05']}"
        )

    all_results.extend(
        criterion_results
    )


# ============================================================
# 8. SIGNIFICANT POST-HOC RESULTS SUMMARY
# ============================================================

significant_results = [
    result
    for result in all_results
    if result["significant_holm_0.05"]
]


print()
print("=" * 78)
print("SIGNIFICANT POST-HOC COMPARISONS")
print("=" * 78)

print()

if not significant_results:

    print(
        "No pairwise comparison remained "
        "significant after Holm correction."
    )

else:

    for result in significant_results:

        print(
            f"{result['criterion']}: "
            f"{result['generator_a']} vs "
            f"{result['generator_b']} | "
            f"Holm p="
            f"{result['holm_p']:.6f} | "
            f"r_rb="
            f"{result['rank_biserial']:+.3f}"
        )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fieldnames = [
    "criterion",
    "generator_a",
    "generator_b",
    "n_nonzero",
    "mean_difference_A_minus_B",
    "wilcoxon_statistic",
    "raw_p",
    "holm_p",
    "rank_biserial",
    "significant_holm_0.05",
]


with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(
        all_results
    )


# ============================================================
# 10. FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("STEP 6B COMPLETE")
print("=" * 78)

print()
print("Created:")
print(OUTPUT_FILE)

print()
print("Multiple-testing strategy:")
print(
    "  Stage 1: Holm correction across "
    "8 criterion-level Friedman tests."
)
print(
    "  Stage 2: For each criterion that "
    "passed Stage 1, Holm correction "
    "across its 6 pairwise comparisons."
)

print()
print(
    f"Significant post-hoc comparisons: "
    f"{len(significant_results)}"
)

print()
print("No API requests were made.")
print("No evaluation data were modified.")

print()
print(
    "Send the complete console output "
    "before we interpret RQ4."
)

print("=" * 78)