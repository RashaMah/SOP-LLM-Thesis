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
    / "self_evaluation_bias_analysis.csv"
)

# Judges that also have a corresponding generator.
# Gemini was not one of the four generators.
SELF_JUDGE_PAIRS = {
    "GPT": "GPT",
    "CLAUDE": "CLAUDE",
}

GENERATORS = [
    "QWEN",
    "LLAMA",
    "GPT",
    "CLAUDE",
]

SCENARIOS = [
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

print("=" * 78)
print("STEP 7 — SELF-EVALUATION BIAS ANALYSIS")
print("=" * 78)

print()
print("Loaded evaluations:", len(rows))
print("Self-evaluable judge families: GPT, CLAUDE")


# ============================================================
# 3. DETECT COLUMN NAMES
# ============================================================

required_candidates = {
    "judge": [
        "judge",
        "judge_model",
    ],
    "generator": [
        "generator",
        "generator_model",
    ],
    "scenario": [
        "scenario_id",
        "scenario",
    ],
    "total": [
        "total_score",
        "total",
    ],
}


def find_column(options):
    for option in options:
        if option in rows[0]:
            return option

    raise ValueError(
        "Could not find any of these columns: "
        + ", ".join(options)
    )


JUDGE_COL = find_column(
    required_candidates["judge"]
)

GENERATOR_COL = find_column(
    required_candidates["generator"]
)

SCENARIO_COL = find_column(
    required_candidates["scenario"]
)

TOTAL_COL = find_column(
    required_candidates["total"]
)

print()
print("Detected columns:")
print(f"  Judge:      {JUDGE_COL}")
print(f"  Generator:  {GENERATOR_COL}")
print(f"  Scenario:   {SCENARIO_COL}")
print(f"  Total:      {TOTAL_COL}")


# ============================================================
# 4. NORMALIZE LABELS
# ============================================================

def normalize_label(value):
    value = value.strip().upper()

    if "GPT" in value:
        return "GPT"

    if "CLAUDE" in value:
        return "CLAUDE"

    if "QWEN" in value:
        return "QWEN"

    if "LLAMA" in value:
        return "LLAMA"

    if "GEMINI" in value:
        return "GEMINI"

    return value


normalized = []

for row in rows:

    normalized.append({
        "judge":
            normalize_label(
                row[JUDGE_COL]
            ),

        "generator":
            normalize_label(
                row[GENERATOR_COL]
            ),

        "scenario":
            row[SCENARIO_COL].strip(),

        "total":
            float(
                row[TOTAL_COL]
            ),
    })


# ============================================================
# 5. VERIFY COMPLETE JUDGE × GENERATOR × SCENARIO DESIGN
# ============================================================

lookup = {}

for row in normalized:

    key = (
        row["judge"],
        row["generator"],
        row["scenario"],
    )

    if key in lookup:
        raise ValueError(
            f"Duplicate evaluation: {key}"
        )

    lookup[key] = row["total"]


for judge in ["GPT", "CLAUDE", "GEMINI"]:

    for generator in GENERATORS:

        for scenario in SCENARIOS:

            key = (
                judge,
                generator,
                scenario,
            )

            if key not in lookup:
                raise ValueError(
                    f"Missing evaluation: {key}"
                )


print()
print("Design verified:")
print("  Judges:       3")
print("  Generators:   4")
print("  Scenarios:   12")
print("  Evaluations: 144")


# ============================================================
# 6. DESCRIPTIVE JUDGE × GENERATOR MATRIX
# ============================================================

print()
print("=" * 78)
print("A. MEAN TOTAL SCORE BY JUDGE × GENERATOR")
print("=" * 78)

print()
print(
    f"{'Judge':<10}"
    f"{'QWEN':>10}"
    f"{'LLAMA':>10}"
    f"{'GPT':>10}"
    f"{'CLAUDE':>10}"
)

print("-" * 50)

for judge in [
    "GPT",
    "CLAUDE",
    "GEMINI",
]:

    means = []

    for generator in GENERATORS:

        values = [
            lookup[
                (
                    judge,
                    generator,
                    scenario,
                )
            ]
            for scenario in SCENARIOS
        ]

        means.append(
            statistics.mean(values)
        )

    print(
        f"{judge:<10}"
        + "".join(
            f"{value:>10.3f}"
            for value in means
        )
    )


# ============================================================
# 7. SELF-FAMILY CONTRAST
# ============================================================
#
# For each self-evaluable judge:
#
# self score =
#   judge's score for own generator
#
# other-family reference =
#   for the SAME scenario, average that judge's
#   scores for the other three generators
#
# This gives 12 matched scenario-level contrasts.
# ============================================================

print()
print("=" * 78)
print("B. SELF-FAMILY vs OTHER-GENERATOR CONTRAST")
print("=" * 78)

results = []


for judge, own_generator in SELF_JUDGE_PAIRS.items():

    self_scores = []
    other_means = []
    differences = []

    for scenario in SCENARIOS:

        self_score = lookup[
            (
                judge,
                own_generator,
                scenario,
            )
        ]

        other_scores = [
            lookup[
                (
                    judge,
                    generator,
                    scenario,
                )
            ]
            for generator in GENERATORS
            if generator != own_generator
        ]

        other_mean = statistics.mean(
            other_scores
        )

        difference = (
            self_score - other_mean
        )

        self_scores.append(
            self_score
        )

        other_means.append(
            other_mean
        )

        differences.append(
            difference
        )

    mean_self = statistics.mean(
        self_scores
    )

    mean_other = statistics.mean(
        other_means
    )

    mean_difference = statistics.mean(
        differences
    )

    median_difference = statistics.median(
        differences
    )

    nonzero_count = sum(
        abs(d) > 1e-12
        for d in differences
    )

    if nonzero_count == 0:

        statistic = 0.0
        raw_p = 1.0

    else:

        test = wilcoxon(
            self_scores,
            other_means,
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
    print(judge)
    print("-" * 50)

    print(
        f"Own-generator mean:       "
        f"{mean_self:.3f}"
    )

    print(
        f"Other-generators mean:    "
        f"{mean_other:.3f}"
    )

    print(
        f"Mean self - others:       "
        f"{mean_difference:+.3f}"
    )

    print(
        f"Median self - others:     "
        f"{median_difference:+.3f}"
    )

    print(
        f"Non-zero scenario pairs:  "
        f"{nonzero_count}"
    )

    print(
        f"Wilcoxon statistic:       "
        f"{statistic:.4f}"
    )

    print(
        f"Raw p:                    "
        f"{raw_p:.8f}"
    )

    print()
    print("Scenario-level contrasts:")

    for scenario, difference in zip(
        SCENARIOS,
        differences
    ):

        print(
            f"  {scenario}: "
            f"{difference:+.3f}"
        )

    results.append({
        "judge":
            judge,

        "own_generator":
            own_generator,

        "n_scenarios":
            12,

        "own_generator_mean":
            mean_self,

        "other_generators_mean":
            mean_other,

        "mean_self_minus_others":
            mean_difference,

        "median_self_minus_others":
            median_difference,

        "nonzero_pairs":
            nonzero_count,

        "wilcoxon_statistic":
            statistic,

        "raw_p":
            raw_p,
    })


# ============================================================
# 8. HOLM CORRECTION FOR TWO SELF-BIAS TESTS
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


print()
print("=" * 78)
print("C. HOLM-CORRECTED SELF-FAMILY TESTS")
print("=" * 78)

print()

for result in results:

    print(
        f"{result['judge']}: "
        f"self - others = "
        f"{result['mean_self_minus_others']:+.3f}, "
        f"raw p={result['raw_p']:.8f}, "
        f"Holm p={result['holm_p']:.8f}, "
        f"significant="
        f"{result['significant_holm_0.05']}"
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fieldnames = [
    "judge",
    "own_generator",
    "n_scenarios",
    "own_generator_mean",
    "other_generators_mean",
    "mean_self_minus_others",
    "median_self_minus_others",
    "nonzero_pairs",
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
# 10. FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("STEP 7 COMPLETE")
print("=" * 78)

print()
print("Created:")
print(OUTPUT_FILE)

print()
print("Interpretation rule:")
print(
    "  Positive self-minus-others values indicate "
    "that the judge assigned higher scores to its "
    "own generator family than to the average of "
    "the other generators."
)

print()
print(
    "  This is a diagnostic association only. "
    "It does not by itself establish causal "
    "self-preference, because generator quality "
    "also differs."
)

print()
print("No API requests were made.")
print("No evaluation data were modified.")

print()
print(
    "Send the complete console output before "
    "we interpret the result."
)

print("=" * 78)