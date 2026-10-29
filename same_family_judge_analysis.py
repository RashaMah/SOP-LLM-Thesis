import csv
import statistics
from pathlib import Path

from scipy.stats import wilcoxon


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
    / "same_family_judge_analysis.csv"
)

SCENARIOS = [
    "GR-A", "GR-B",
    "RL-A", "RL-B",
    "IR-A", "IR-B",
    "OP-A", "OP-B",
    "ST-A", "ST-B",
    "PD-A", "PD-B",
]

# Only GPT and Claude have both a generator
# and a corresponding judge in this experiment.
FAMILIES = [
    "GPT",
    "CLAUDE",
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
        f"Expected 144 evaluations, found {len(rows)}."
    )


# ============================================================
# 3. DETECT COLUMNS
# ============================================================

def find_column(options):
    for option in options:
        if option in rows[0]:
            return option

    raise ValueError(
        "Could not find any of these columns: "
        + ", ".join(options)
    )


JUDGE_COL = find_column(
    ["judge", "judge_model"]
)

GENERATOR_COL = find_column(
    ["generator", "generator_model"]
)

SCENARIO_COL = find_column(
    ["scenario_id", "scenario"]
)

TOTAL_COL = find_column(
    ["total_score", "total"]
)


# ============================================================
# 4. NORMALIZE LABELS
# ============================================================

def normalize_label(value):

    value = value.strip().upper()

    if "GPT" in value:
        return "GPT"

    if "CLAUDE" in value:
        return "CLAUDE"

    if "GEMINI" in value:
        return "GEMINI"

    if "QWEN" in value:
        return "QWEN"

    if "LLAMA" in value:
        return "LLAMA"

    return value


lookup = {}

for row in rows:

    judge = normalize_label(
        row[JUDGE_COL]
    )

    generator = normalize_label(
        row[GENERATOR_COL]
    )

    scenario = row[
        SCENARIO_COL
    ].strip()

    total = float(
        row[TOTAL_COL]
    )

    key = (
        judge,
        generator,
        scenario,
    )

    if key in lookup:
        raise ValueError(
            f"Duplicate evaluation: {key}"
        )

    lookup[key] = total


# ============================================================
# 5. VERIFY REQUIRED DATA
# ============================================================

for family in FAMILIES:

    for scenario in SCENARIOS:

        for judge in [
            "GPT",
            "CLAUDE",
            "GEMINI",
        ]:

            key = (
                judge,
                family,
                scenario,
            )

            if key not in lookup:
                raise ValueError(
                    f"Missing evaluation: {key}"
                )


print("=" * 78)
print("STEP 7B — SAME-FAMILY JUDGE DIAGNOSTIC")
print("=" * 78)

print()
print("Design verified:")
print("  Generator families tested:  GPT, CLAUDE")
print("  Scenarios per family:        12")
print("  Judges per SOP:               3")


# ============================================================
# 6. ANALYSIS
# ============================================================
#
# For each generator family:
#
# Same-family score =
#   GPT judge → GPT-generated SOP
#   Claude judge → Claude-generated SOP
#
# Independent reference =
#   mean score assigned to those exact same SOPs
#   by the OTHER TWO judges.
#
# Thus generator quality is held substantially
# more constant than in the Step 7 diagnostic.
# ============================================================

results = []


for family in FAMILIES:

    same_family_scores = []
    independent_means = []
    differences = []

    print()
    print("=" * 78)
    print(f"{family}-GENERATED SOPs")
    print("=" * 78)

    print()
    print(
        f"{'Scenario':<10}"
        f"{'Same-family':>15}"
        f"{'Other judges':>15}"
        f"{'Difference':>15}"
    )

    print("-" * 55)

    for scenario in SCENARIOS:

        same_family_score = lookup[
            (
                family,
                family,
                scenario,
            )
        ]

        other_judges = [
            judge
            for judge in [
                "GPT",
                "CLAUDE",
                "GEMINI",
            ]
            if judge != family
        ]

        other_scores = [
            lookup[
                (
                    judge,
                    family,
                    scenario,
                )
            ]
            for judge in other_judges
        ]

        independent_mean = (
            statistics.mean(
                other_scores
            )
        )

        difference = (
            same_family_score
            - independent_mean
        )

        same_family_scores.append(
            same_family_score
        )

        independent_means.append(
            independent_mean
        )

        differences.append(
            difference
        )

        print(
            f"{scenario:<10}"
            f"{same_family_score:>15.3f}"
            f"{independent_mean:>15.3f}"
            f"{difference:>+15.3f}"
        )


    mean_same = statistics.mean(
        same_family_scores
    )

    mean_independent = statistics.mean(
        independent_means
    )

    mean_difference = statistics.mean(
        differences
    )

    median_difference = statistics.median(
        differences
    )

    positive = sum(
        d > 1e-12
        for d in differences
    )

    negative = sum(
        d < -1e-12
        for d in differences
    )

    zero = len(differences) - positive - negative

    nonzero = positive + negative


    # --------------------------------------------------------
    # Wilcoxon
    # --------------------------------------------------------

    if nonzero == 0:

        statistic = 0.0
        raw_p = 1.0

    else:

        test = wilcoxon(
            same_family_scores,
            independent_means,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )

        statistic = float(
            test.statistic
        )

        raw_p = float(
            test.pvalue
        )


    print()
    print("Summary")
    print("-" * 55)

    print(
        f"Same-family judge mean:    "
        f"{mean_same:.3f}"
    )

    print(
        f"Other-two-judges mean:     "
        f"{mean_independent:.3f}"
    )

    print(
        f"Mean difference:           "
        f"{mean_difference:+.3f}"
    )

    print(
        f"Median difference:         "
        f"{median_difference:+.3f}"
    )

    print(
        f"Positive / zero / negative:"
        f" {positive} / {zero} / {negative}"
    )

    print(
        f"Wilcoxon statistic:        "
        f"{statistic:.4f}"
    )

    print(
        f"Raw p:                     "
        f"{raw_p:.8f}"
    )


    results.append({
        "generator_family":
            family,

        "same_family_judge":
            family,

        "n_scenarios":
            12,

        "same_family_mean":
            mean_same,

        "other_two_judges_mean":
            mean_independent,

        "mean_difference":
            mean_difference,

        "median_difference":
            median_difference,

        "positive_differences":
            positive,

        "zero_differences":
            zero,

        "negative_differences":
            negative,

        "wilcoxon_statistic":
            statistic,

        "raw_p":
            raw_p,
    })


# ============================================================
# 7. HOLM CORRECTION ACROSS THE TWO TESTS
# ============================================================

ordered = sorted(
    range(len(results)),
    key=lambda i:
        results[i]["raw_p"]
)

previous_adjusted = 0.0
m = len(results)

for position, index in enumerate(
    ordered
):

    adjusted = (
        results[index]["raw_p"]
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

    results[index][
        "holm_p"
    ] = adjusted

    results[index][
        "significant_holm_0.05"
    ] = (
        adjusted < 0.05
    )

    previous_adjusted = adjusted


# ============================================================
# 8. FINAL COMPARISON
# ============================================================

print()
print("=" * 78)
print("HOLM-CORRECTED RESULTS")
print("=" * 78)

print()

for result in results:

    print(
        f"{result['generator_family']}: "
        f"same-family minus other judges = "
        f"{result['mean_difference']:+.3f}, "
        f"raw p={result['raw_p']:.8f}, "
        f"Holm p={result['holm_p']:.8f}, "
        f"significant="
        f"{result['significant_holm_0.05']}"
    )


# ============================================================
# 9. SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fieldnames = [
    "generator_family",
    "same_family_judge",
    "n_scenarios",
    "same_family_mean",
    "other_two_judges_mean",
    "mean_difference",
    "median_difference",
    "positive_differences",
    "zero_differences",
    "negative_differences",
    "wilcoxon_statistic",
    "raw_p",
    "holm_p",
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
        results
    )


# ============================================================
# 10. COMPLETE
# ============================================================

print()
print("=" * 78)
print("STEP 7B COMPLETE")
print("=" * 78)

print()
print("Created:")
print(OUTPUT_FILE)

print()
print(
    "Positive differences mean that the "
    "same-family judge scored its generator's "
    "SOPs higher than the mean of the other "
    "two judges scoring those exact same SOPs."
)

print()
print(
    "This is a stronger self-preference "
    "diagnostic than Step 7, but it still "
    "does not establish causality."
)

print()
print("No API requests were made.")
print("No evaluation data were modified.")

print()
print(
    "Send the complete console output "
    "before we freeze Step 7."
)

print("=" * 78)

#A same-family scoring diagnostic revealed asymmetric judge effects. For GPT-generated SOPs, the GPT judge assigned significantly lower scores than the mean of the two other judges (mean difference = −1.625, Holm-adjusted p = .0156). In contrast, for Claude-generated SOPs, the Claude judge assigned significantly higher scores than the other judges (mean difference = +0.958, Holm-adjusted p = .0186). Thus, the results do not indicate a uniform same-family preference across evaluators, but they demonstrate that judge identity can interact systematically with generator identity.