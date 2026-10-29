import csv
from pathlib import Path

from scipy.stats import friedmanchisquare


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
    / "criterion_level_analysis.csv"
)

GENERATORS = [
    "QWEN",
    "LLAMA",
    "GPT",
    "CLAUDE",
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

    criterion_scores = {}

    for criterion in CRITERIA:

        column = f"{criterion}_mean"

        if column not in row:
            raise ValueError(
                f"Missing column: {column}"
            )

        score = float(
            row[column]
        )

        if not 1 <= score <= 5:
            raise ValueError(
                f"Invalid score: "
                f"{scenario} / "
                f"{generator} / "
                f"{criterion}"
            )

        criterion_scores[
            criterion
        ] = score

    data[key] = criterion_scores


# Verify complete matched design

for scenario in EXPECTED_SCENARIOS:

    for generator in GENERATORS:

        if (
            scenario,
            generator
        ) not in data:

            raise ValueError(
                f"Missing: "
                f"{scenario} / {generator}"
            )


print("=" * 78)
print("STEP 6 — CRITERION-LEVEL GENERATOR ANALYSIS")
print("=" * 78)

print()
print("Design verified:")
print("  Scenarios:              12")
print("  Generators:              4")
print("  Criteria:                8")
print("  SOPs:                   48")


# ============================================================
# 4. DESCRIPTIVE CRITERION STATISTICS
# ============================================================

print()
print("=" * 78)
print("A. CRITERION MEANS BY GENERATOR")
print("=" * 78)

descriptive = {}

for criterion in CRITERIA:

    descriptive[
        criterion
    ] = {}

    print()
    print(criterion)
    print("-" * 55)

    for generator in GENERATORS:

        values = [
            data[
                (scenario, generator)
            ][criterion]
            for scenario
            in EXPECTED_SCENARIOS
        ]

        mean_value = (
            sum(values)
            / len(values)
        )

        minimum = min(values)
        maximum = max(values)

        descriptive[
            criterion
        ][generator] = {
            "mean": mean_value,
            "min": minimum,
            "max": maximum,
        }

        print(
            f"{generator:<8} "
            f"Mean={mean_value:.3f}   "
            f"Min={minimum:.3f}   "
            f"Max={maximum:.3f}"
        )


# ============================================================
# 5. FRIEDMAN TEST FOR EACH CRITERION
# ============================================================

print()
print("=" * 78)
print("B. FRIEDMAN TESTS BY CRITERION")
print("=" * 78)

criterion_tests = []


for criterion in CRITERIA:

    generator_scores = []

    for generator in GENERATORS:

        values = [
            data[
                (scenario, generator)
            ][criterion]
            for scenario
            in EXPECTED_SCENARIOS
        ]

        generator_scores.append(
            values
        )

    result = friedmanchisquare(
        *generator_scores
    )

    statistic = float(
        result.statistic
    )

    raw_p = float(
        result.pvalue
    )

    kendalls_w = (
        statistic
        / (
            len(EXPECTED_SCENARIOS)
            * (
                len(GENERATORS) - 1
            )
        )
    )

    criterion_tests.append({
        "criterion":
            criterion,

        "friedman_chi_square":
            statistic,

        "df":
            3,

        "raw_p":
            raw_p,

        "kendalls_w":
            kendalls_w,
    })


# ============================================================
# 6. HOLM CORRECTION ACROSS 8 CRITERIA
# ============================================================

ordered = sorted(
    range(
        len(criterion_tests)
    ),
    key=lambda i:
        criterion_tests[i][
            "raw_p"
        ]
)

m = len(
    criterion_tests
)

previous_adjusted = 0.0


for position, index in enumerate(
    ordered
):

    raw_p = criterion_tests[
        index
    ]["raw_p"]

    adjusted = (
        raw_p
        * (
            m - position
        )
    )

    adjusted = max(
        adjusted,
        previous_adjusted
    )

    adjusted = min(
        adjusted,
        1.0
    )

    criterion_tests[
        index
    ]["holm_p"] = adjusted

    criterion_tests[
        index
    ]["significant_holm_0.05"] = (
        adjusted < ALPHA
    )

    previous_adjusted = adjusted


# ============================================================
# 7. PRINT TEST RESULTS
# ============================================================

print()

for result in criterion_tests:

    print(
        result["criterion"]
    )

    print("-" * 55)

    print(
        f"Friedman chi-square: "
        f"{result['friedman_chi_square']:.6f}"
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
        f"Kendall's W:         "
        f"{result['kendalls_w']:.4f}"
    )

    print(
        f"Significant (Holm):  "
        f"{result['significant_holm_0.05']}"
    )

    print()


# ============================================================
# 8. OVERALL CRITERION PERFORMANCE
# ============================================================
#
# Average across all 48 SOPs.
#
# This is descriptive and helps identify dimensions
# with lower scores across the experiment.
# ============================================================

print("=" * 78)
print("C. OVERALL CRITERION PERFORMANCE")
print("=" * 78)

overall_criteria = []


for criterion in CRITERIA:

    values = []

    for scenario in EXPECTED_SCENARIOS:

        for generator in GENERATORS:

            values.append(
                data[
                    (scenario, generator)
                ][criterion]
            )

    mean_value = (
        sum(values)
        / len(values)
    )

    below_five = sum(
        value < 5.0
        for value in values
    )

    overall_criteria.append({
        "criterion":
            criterion,

        "overall_mean":
            mean_value,

        "below_5_count":
            below_five,

        "n":
            len(values),
    })


# Sort ONLY for descriptive inspection:
# lowest mean first.

sorted_overall = sorted(
    overall_criteria,
    key=lambda x:
        x["overall_mean"]
)


print()

for result in sorted_overall:

    print(
        f"{result['criterion']}: "
        f"Mean={result['overall_mean']:.3f}, "
        f"<5={result['below_5_count']}/"
        f"{result['n']}"
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_rows = []


for test in criterion_tests:

    criterion = test[
        "criterion"
    ]

    row = {
        "criterion":
            criterion,

        "friedman_chi_square":
            test[
                "friedman_chi_square"
            ],

        "df":
            test["df"],

        "raw_p":
            test["raw_p"],

        "holm_p":
            test["holm_p"],

        "kendalls_w":
            test["kendalls_w"],

        "significant_holm_0.05":
            test[
                "significant_holm_0.05"
            ],
    }

    for generator in GENERATORS:

        row[
            f"{generator}_mean"
        ] = descriptive[
            criterion
        ][generator]["mean"]

    overall_match = next(
        item
        for item in overall_criteria
        if item["criterion"]
        == criterion
    )

    row[
        "overall_mean"
    ] = overall_match[
        "overall_mean"
    ]

    row[
        "below_5_count"
    ] = overall_match[
        "below_5_count"
    ]

    output_rows.append(
        row
    )


fieldnames = [
    "criterion",
    "friedman_chi_square",
    "df",
    "raw_p",
    "holm_p",
    "kendalls_w",
    "significant_holm_0.05",
    "QWEN_mean",
    "LLAMA_mean",
    "GPT_mean",
    "CLAUDE_mean",
    "overall_mean",
    "below_5_count",
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
        output_rows
    )


# ============================================================
# 10. FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("STEP 6 COMPLETE")
print("=" * 78)

print()
print("Created:")
print(OUTPUT_FILE)

print()
print("Statistical design:")
print(
    "  8 criterion-specific Friedman tests"
)
print(
    "  Holm correction across 8 omnibus tests"
)
print(
    "  Kendall's W reported as effect size"
)

print()
print(
    "Criterion ordering in Section C is "
    "descriptive only."
)

print()
print("No API requests were made.")
print("No evaluation data were modified.")

print()
print(
    "Send the complete console output "
    "before we perform criterion-specific "
    "post-hoc comparisons."
)

print("=" * 78)