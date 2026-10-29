import csv
import itertools
from pathlib import Path

import scipy
from scipy.stats import friedmanchisquare, wilcoxon


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

GENERATORS = [
    "QWEN",
    "LLAMA",
    "GPT",
    "CLAUDE",
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
# 3. BUILD MATCHED MATRIX
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

    if scenario not in data:
        data[scenario] = {}

    if generator in data[scenario]:
        raise ValueError(
            f"Duplicate: {scenario} / {generator}"
        )

    data[scenario][generator] = float(
        row["consensus_total_mean"]
    )


for scenario in EXPECTED_SCENARIOS:

    if scenario not in data:
        raise ValueError(
            f"Missing scenario: {scenario}"
        )

    if set(data[scenario]) != set(GENERATORS):
        raise ValueError(
            f"Incomplete generator data: {scenario}"
        )


scores = {
    generator: [
        data[scenario][generator]
        for scenario in EXPECTED_SCENARIOS
    ]
    for generator in GENERATORS
}


# ============================================================
# 4. REPORT SOFTWARE VERSION
# ============================================================

print("=" * 78)
print("STEP 4A — INDEPENDENT SCIPY VALIDATION")
print("=" * 78)

print()
print(f"SciPy version: {scipy.__version__}")

print()
print("Dataset verified:")
print("  Matched scenarios:       12")
print("  Generator models:         4")
print("  Total SOPs:              48")


# ============================================================
# 5. SCIPY FRIEDMAN TEST
# ============================================================

friedman_result = friedmanchisquare(
    scores["QWEN"],
    scores["LLAMA"],
    scores["GPT"],
    scores["CLAUDE"],
)

friedman_stat = float(
    friedman_result.statistic
)

friedman_p = float(
    friedman_result.pvalue
)

kendalls_w = (
    friedman_stat
    / (
        len(EXPECTED_SCENARIOS)
        * (
            len(GENERATORS) - 1
        )
    )
)


print()
print("=" * 78)
print("A. SCIPY FRIEDMAN TEST")
print("=" * 78)

print()
print(
    f"Friedman chi-square: "
    f"{friedman_stat:.8f}"
)

print(
    f"Degrees of freedom:  "
    f"{len(GENERATORS) - 1}"
)

print(
    f"p-value:             "
    f"{friedman_p:.10f}"
)

print(
    f"Kendall's W:         "
    f"{kendalls_w:.8f}"
)


# ============================================================
# 6. SCIPY WILCOXON TESTS
# ============================================================

pairwise_results = []


print()
print("=" * 78)
print("B. SCIPY PAIRED WILCOXON TESTS")
print("=" * 78)

print()
print(
    "method='auto', zero_method='wilcox', "
    "two-sided"
)
print()


for generator_a, generator_b in itertools.combinations(
    GENERATORS,
    2
):

    x = scores[generator_a]
    y = scores[generator_b]

    differences = [
        a - b
        for a, b in zip(x, y)
    ]

    nonzero = [
        d for d in differences
        if abs(d) > 1e-12
    ]

    result = wilcoxon(
        x,
        y,
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

    # Matched-pairs rank-biserial correlation.
    #
    # Rank the absolute non-zero differences,
    # using average ranks for ties.

    absolute_values = [
        abs(d)
        for d in nonzero
    ]

    sorted_indices = sorted(
        range(len(absolute_values)),
        key=lambda i: absolute_values[i]
    )

    ranks = [0.0] * len(
        absolute_values
    )

    i = 0

    while i < len(sorted_indices):

        j = i

        while (
            j + 1 < len(sorted_indices)
            and abs(
                absolute_values[
                    sorted_indices[j + 1]
                ]
                - absolute_values[
                    sorted_indices[i]
                ]
            ) < 1e-12
        ):
            j += 1

        average_rank = (
            (i + 1) + (j + 1)
        ) / 2.0

        for position in range(
            i,
            j + 1
        ):
            ranks[
                sorted_indices[position]
            ] = average_rank

        i = j + 1

    w_plus = sum(
        rank
        for d, rank in zip(
            nonzero,
            ranks
        )
        if d > 0
    )

    w_minus = sum(
        rank
        for d, rank in zip(
            nonzero,
            ranks
        )
        if d < 0
    )

    rank_total = (
        w_plus + w_minus
    )

    if rank_total == 0:
        rank_biserial = 0.0
    else:
        rank_biserial = (
            w_plus - w_minus
        ) / rank_total

    pairwise_results.append({
        "a": generator_a,
        "b": generator_b,
        "n_nonzero": len(nonzero),
        "statistic": statistic,
        "raw_p": raw_p,
        "rank_biserial": rank_biserial,
    })


# ============================================================
# 7. HOLM MULTIPLE-COMPARISON CORRECTION
# ============================================================

m = len(pairwise_results)

ordered = sorted(
    range(m),
    key=lambda idx:
        pairwise_results[idx]["raw_p"]
)

previous_adjusted = 0.0

for position, idx in enumerate(ordered):

    raw_p = pairwise_results[
        idx
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

    pairwise_results[
        idx
    ]["holm_p"] = adjusted

    pairwise_results[
        idx
    ]["significant"] = (
        adjusted < ALPHA
    )

    previous_adjusted = adjusted


# ============================================================
# 8. PRINT VALIDATED PAIRWISE RESULTS
# ============================================================

for result in pairwise_results:

    print(
        f"{result['a']} vs "
        f"{result['b']}"
    )

    print("-" * 45)

    print(
        f"Non-zero pairs:      "
        f"{result['n_nonzero']}"
    )

    print(
        f"SciPy statistic:     "
        f"{result['statistic']:.4f}"
    )

    print(
        f"Raw p:               "
        f"{result['raw_p']:.10f}"
    )

    print(
        f"Holm-adjusted p:     "
        f"{result['holm_p']:.10f}"
    )

    print(
        f"Rank-biserial r:     "
        f"{result['rank_biserial']:.4f}"
    )

    print(
        f"Significant (Holm):  "
        f"{result['significant']}"
    )

    print()


# ============================================================
# 9. COMPARE WITH ORIGINAL STEP 4 VALUES
# ============================================================

print("=" * 78)
print("C. ORIGINAL STEP 4 REFERENCE VALUES")
print("=" * 78)

print()
print(
    "Original Friedman chi-square: "
    "21.540541"
)
print(
    "Original Friedman p:          "
    "0.00008129"
)
print(
    "Original Kendall's W:         "
    "0.598348"
)

print()
print(
    "This section is printed only "
    "for direct visual comparison."
)


# ============================================================
# 10. FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("STEP 4A VALIDATION COMPLETE")
print("=" * 78)

print()
print(
    "No API requests were made."
)
print(
    "No source data or evaluation "
    "results were modified."
)

print()
print(
    "Send the complete console output "
    "before we freeze the inferential "
    "results."
)

print("=" * 78)

#For the thesis, we should use the SciPy-validated values, not the earlier hand-calculated approximate Wilcoxon p-values. The earlier script can remain in the audit trail; it simply isn't the final inferential result