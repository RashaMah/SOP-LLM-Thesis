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
    / "variant_A_vs_B_analysis.csv"
)

GENERATORS = [
    "QWEN",
    "LLAMA",
    "GPT",
    "CLAUDE",
]

PROCESSES = [
    "GR",
    "RL",
    "IR",
    "OP",
    "ST",
    "PD",
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

if len(rows) != 48:
    raise ValueError(
        f"Expected 48 rows, found {len(rows)}."
    )


# ============================================================
# 3. BUILD GENERATOR × PROCESS × VARIANT DATA
# ============================================================

data = {}

for row in rows:

    generator = row["generator"]
    scenario_id = row["scenario_id"]
    variant = row["variant"]

    if generator not in GENERATORS:
        raise ValueError(
            f"Unexpected generator: {generator}"
        )

    if variant not in ["A", "B"]:
        raise ValueError(
            f"Unexpected variant: {variant}"
        )

    process = scenario_id.split("-")[0]

    if process not in PROCESSES:
        raise ValueError(
            f"Unexpected process: {process}"
        )

    key = (
        generator,
        process,
        variant,
    )

    if key in data:
        raise ValueError(
            f"Duplicate entry: {key}"
        )

    data[key] = float(
        row["consensus_total_mean"]
    )


# Verify complete design

for generator in GENERATORS:
    for process in PROCESSES:
        for variant in ["A", "B"]:

            key = (
                generator,
                process,
                variant,
            )

            if key not in data:
                raise ValueError(
                    f"Missing: {key}"
                )


print("=" * 78)
print("STEP 5 — VARIANT A vs B ANALYSIS")
print("=" * 78)

print()
print("Design verified:")
print("  Generators:             4")
print("  Process families:       6")
print("  Variants per process:   2")
print("  SOPs:                  48")


# ============================================================
# 4. DESCRIPTIVE A/B RESULTS BY GENERATOR
# ============================================================

print()
print("=" * 78)
print("A. A vs B DESCRIPTIVE RESULTS BY GENERATOR")
print("=" * 78)

results = []


for generator in GENERATORS:

    scores_a = [
        data[
            (
                generator,
                process,
                "A",
            )
        ]
        for process in PROCESSES
    ]

    scores_b = [
        data[
            (
                generator,
                process,
                "B",
            )
        ]
        for process in PROCESSES
    ]

    differences = [
        b - a
        for a, b in zip(
            scores_a,
            scores_b
        )
    ]

    mean_a = statistics.mean(
        scores_a
    )

    mean_b = statistics.mean(
        scores_b
    )

    mean_difference = statistics.mean(
        differences
    )

    median_difference = statistics.median(
        differences
    )

    print()
    print(generator)
    print("-" * 45)

    print(
        f"Variant A mean:      "
        f"{mean_a:.3f}"
    )

    print(
        f"Variant B mean:      "
        f"{mean_b:.3f}"
    )

    print(
        f"Mean B - A:          "
        f"{mean_difference:.3f}"
    )

    print(
        f"Median B - A:        "
        f"{median_difference:.3f}"
    )

    print()
    print("Process-level differences (B - A):")

    for process, difference in zip(
        PROCESSES,
        differences
    ):
        print(
            f"  {process}: "
            f"{difference:+.3f}"
        )

    results.append({
        "analysis":
            "generator_descriptive",

        "generator":
            generator,

        "n_pairs":
            6,

        "mean_A":
            mean_a,

        "mean_B":
            mean_b,

        "mean_B_minus_A":
            mean_difference,

        "median_B_minus_A":
            median_difference,

        "wilcoxon_statistic":
            "",

        "raw_p":
            "",

        "holm_p":
            "",

        "significant_holm_0.05":
            "",
    })


# ============================================================
# 5. GENERATOR-SPECIFIC WILCOXON TESTS
# ============================================================

print()
print("=" * 78)
print("B. GENERATOR-SPECIFIC PAIRED WILCOXON TESTS")
print("=" * 78)

print()
print(
    "Each test contains only six matched "
    "process pairs. Interpret cautiously."
)
print()


generator_tests = []


for generator in GENERATORS:

    scores_a = [
        data[
            (
                generator,
                process,
                "A",
            )
        ]
        for process in PROCESSES
    ]

    scores_b = [
        data[
            (
                generator,
                process,
                "B",
            )
        ]
        for process in PROCESSES
    ]

    differences = [
        b - a
        for a, b in zip(
            scores_a,
            scores_b
        )
    ]

    nonzero_count = sum(
        abs(d) > 1e-12
        for d in differences
    )

    if nonzero_count == 0:

        statistic = 0.0
        raw_p = 1.0

    else:

        test = wilcoxon(
            scores_b,
            scores_a,
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

    generator_tests.append({
        "generator":
            generator,

        "statistic":
            statistic,

        "raw_p":
            raw_p,

        "nonzero":
            nonzero_count,
    })


# Holm correction across four generator-specific tests

ordered = sorted(
    range(len(generator_tests)),
    key=lambda i:
        generator_tests[i]["raw_p"]
)

previous_adjusted = 0.0
m = len(generator_tests)

for position, index in enumerate(
    ordered
):

    adjusted = (
        generator_tests[index]["raw_p"]
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

    generator_tests[
        index
    ]["holm_p"] = adjusted

    generator_tests[
        index
    ]["significant"] = (
        adjusted < 0.05
    )

    previous_adjusted = adjusted


for test in generator_tests:

    print(test["generator"])
    print("-" * 45)

    print(
        f"Non-zero pairs:      "
        f"{test['nonzero']}"
    )

    print(
        f"Wilcoxon statistic:  "
        f"{test['statistic']:.4f}"
    )

    print(
        f"Raw p:               "
        f"{test['raw_p']:.8f}"
    )

    print(
        f"Holm-adjusted p:     "
        f"{test['holm_p']:.8f}"
    )

    print(
        f"Significant (Holm):  "
        f"{test['significant']}"
    )

    print()

    # Add test information to corresponding
    # descriptive result row.

    for row in results:

        if (
            row["analysis"]
            == "generator_descriptive"
            and row["generator"]
            == test["generator"]
        ):

            row[
                "wilcoxon_statistic"
            ] = test[
                "statistic"
            ]

            row[
                "raw_p"
            ] = test[
                "raw_p"
            ]

            row[
                "holm_p"
            ] = test[
                "holm_p"
            ]

            row[
                "significant_holm_0.05"
            ] = test[
                "significant"
            ]


# ============================================================
# 6. OVERALL PROCESS-LEVEL A/B EFFECT
# ============================================================
#
# To avoid pretending that the four generators
# are 24 independent process observations,
# first average the four generators within each
# process × variant.
#
# This produces SIX independent matched process
# pairs for the overall A/B comparison.
# ============================================================

process_a = []
process_b = []

print()
print("=" * 78)
print("C. OVERALL A vs B — PROCESS-LEVEL AGGREGATION")
print("=" * 78)

print()
print(
    "Four generator scores are averaged "
    "within each process and variant."
)

print()
print(
    f"{'Process':<10}"
    f"{'A mean':>12}"
    f"{'B mean':>12}"
    f"{'B - A':>12}"
)

print("-" * 46)


for process in PROCESSES:

    a_values = [
        data[
            (
                generator,
                process,
                "A",
            )
        ]
        for generator in GENERATORS
    ]

    b_values = [
        data[
            (
                generator,
                process,
                "B",
            )
        ]
        for generator in GENERATORS
    ]

    a_mean = statistics.mean(
        a_values
    )

    b_mean = statistics.mean(
        b_values
    )

    difference = (
        b_mean - a_mean
    )

    process_a.append(
        a_mean
    )

    process_b.append(
        b_mean
    )

    print(
        f"{process:<10}"
        f"{a_mean:>12.3f}"
        f"{b_mean:>12.3f}"
        f"{difference:>+12.3f}"
    )


overall_differences = [
    b - a
    for a, b in zip(
        process_a,
        process_b
    )
]

overall_mean_a = statistics.mean(
    process_a
)

overall_mean_b = statistics.mean(
    process_b
)

overall_mean_difference = (
    statistics.mean(
        overall_differences
    )
)

overall_median_difference = (
    statistics.median(
        overall_differences
    )
)


overall_test = wilcoxon(
    process_b,
    process_a,
    zero_method="wilcox",
    correction=False,
    alternative="two-sided",
    method="auto",
)


print()
print("Overall process-level results:")
print(
    f"Mean A:              "
    f"{overall_mean_a:.3f}"
)

print(
    f"Mean B:              "
    f"{overall_mean_b:.3f}"
)

print(
    f"Mean B - A:          "
    f"{overall_mean_difference:.3f}"
)

print(
    f"Median B - A:        "
    f"{overall_median_difference:.3f}"
)

print(
    f"Wilcoxon statistic:  "
    f"{float(overall_test.statistic):.4f}"
)

print(
    f"Two-sided p:         "
    f"{float(overall_test.pvalue):.8f}"
)


results.append({
    "analysis":
        "overall_process_level",

    "generator":
        "ALL_GENERATORS_AVERAGED",

    "n_pairs":
        6,

    "mean_A":
        overall_mean_a,

    "mean_B":
        overall_mean_b,

    "mean_B_minus_A":
        overall_mean_difference,

    "median_B_minus_A":
        overall_median_difference,

    "wilcoxon_statistic":
        float(
            overall_test.statistic
        ),

    "raw_p":
        float(
            overall_test.pvalue
        ),

    "holm_p":
        "",

    "significant_holm_0.05":
        "",
})


# ============================================================
# 7. SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fields = [
    "analysis",
    "generator",
    "n_pairs",
    "mean_A",
    "mean_B",
    "mean_B_minus_A",
    "median_B_minus_A",
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
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(
        results
    )


# ============================================================
# 8. FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("STEP 5 COMPLETE")
print("=" * 78)

print()
print("Created:")
print(OUTPUT_FILE)

print()
print("Important:")
print(
    "  Generator-specific tests use "
    "only 6 matched process pairs."
)
print(
    "  The overall test also uses "
    "6 process-level matched pairs."
)
print(
    "  Therefore inference must be "
    "interpreted cautiously."
)

print()
print("No API requests were made.")
print("No evaluation data were modified.")

print()
print(
    "Send the complete console output "
    "before interpreting the A/B effect."
)

print("=" * 78)

#Within the six matched process families examined in this study, exception- and constraint-rich Variant B scenarios did not show a consistent overall reduction in consensus SOP quality compared with the corresponding standard Variant A scenarios. The process-level mean difference was negligible (+0.042 points), and the paired comparison was not statistically significant (Wilcoxon signed-rank test, p = 1.000). However, effects varied across individual processes and generators, and the small number of matched process pairs (n = 6) limits inferential power.