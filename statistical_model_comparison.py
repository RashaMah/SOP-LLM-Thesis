import csv
import math
import itertools
from pathlib import Path


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
    / "statistical_generator_comparison.csv"
)

GENERATORS = [
    "QWEN",
    "LLAMA",
    "GPT",
    "CLAUDE"
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
# 2. BASIC STATISTICAL FUNCTIONS
# ============================================================

def normal_cdf(x):
    return 0.5 * (
        1.0 + math.erf(
            x / math.sqrt(2.0)
        )
    )


def chi_square_sf_df3(x):
    """
    Survival function for chi-square with df=3.

    For df=3:
    SF(x) =
    erfc(sqrt(x/2))
    + 2/sqrt(pi) * sqrt(x/2) * exp(-x/2)
    """

    if x < 0:
        return 1.0

    z = math.sqrt(x / 2.0)

    return (
        math.erfc(z)
        + (
            2.0
            / math.sqrt(math.pi)
        )
        * z
        * math.exp(
            -(z ** 2)
        )
    )


def average_ranks(values):
    """
    Rank values from lowest to highest.
    Ties receive average ranks.
    """

    indexed = sorted(
        enumerate(values),
        key=lambda x: x[1]
    )

    ranks = [0.0] * len(values)

    i = 0

    while i < len(indexed):

        j = i

        while (
            j + 1 < len(indexed)
            and indexed[j + 1][1]
            == indexed[i][1]
        ):
            j += 1

        average_rank = (
            (i + 1) + (j + 1)
        ) / 2.0

        for position in range(
            i,
            j + 1
        ):
            original_index = (
                indexed[position][0]
            )

            ranks[
                original_index
            ] = average_rank

        i = j + 1

    return ranks


# ============================================================
# 3. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n"
        f"{INPUT_FILE}"
    )

with INPUT_FILE.open(
    "r",
    newline="",
    encoding="utf-8"
) as f:

    rows = list(
        csv.DictReader(f)
    )


if len(rows) != 48:
    raise ValueError(
        f"Expected 48 SOP rows, "
        f"found {len(rows)}."
    )


# ============================================================
# 4. BUILD MATCHED SCENARIO × GENERATOR MATRIX
# ============================================================

data = {}

for row in rows:

    scenario_id = row["scenario_id"]
    generator = row["generator"]

    if scenario_id not in EXPECTED_SCENARIOS:
        raise ValueError(
            f"Unexpected scenario: "
            f"{scenario_id}"
        )

    if generator not in GENERATORS:
        raise ValueError(
            f"Unexpected generator: "
            f"{generator}"
        )

    if scenario_id not in data:
        data[scenario_id] = {}

    if generator in data[scenario_id]:
        raise ValueError(
            f"Duplicate entry: "
            f"{scenario_id} / {generator}"
        )

    score = float(
        row["consensus_total_mean"]
    )

    if not 8 <= score <= 40:
        raise ValueError(
            f"Invalid consensus score: "
            f"{score}"
        )

    data[scenario_id][
        generator
    ] = score


# Verify complete 12 × 4 matrix

if set(data.keys()) != set(
    EXPECTED_SCENARIOS
):
    raise ValueError(
        "Scenario set is incomplete."
    )

for scenario_id in EXPECTED_SCENARIOS:

    if set(
        data[scenario_id].keys()
    ) != set(GENERATORS):

        raise ValueError(
            f"{scenario_id}: does not "
            f"contain all four generators."
        )


print("=" * 78)
print("STEP 4 — STATISTICAL GENERATOR COMPARISON")
print("=" * 78)

print()
print("Matched design verified:")
print("  Scenarios / blocks:       12")
print("  Generator models:          4")
print("  Matched observations:     48")
print("  Significance level:     0.05")


# ============================================================
# 5. SHOW MATCHED SCORE MATRIX
# ============================================================

print()
print("=" * 78)
print("A. MATCHED CONSENSUS SCORE MATRIX")
print("=" * 78)

print()
print(
    f"{'Scenario':<10}"
    f"{'QWEN':>10}"
    f"{'LLAMA':>10}"
    f"{'GPT':>10}"
    f"{'CLAUDE':>10}"
)

print("-" * 50)

for scenario_id in EXPECTED_SCENARIOS:

    print(
        f"{scenario_id:<10}"
        f"{data[scenario_id]['QWEN']:>10.3f}"
        f"{data[scenario_id]['LLAMA']:>10.3f}"
        f"{data[scenario_id]['GPT']:>10.3f}"
        f"{data[scenario_id]['CLAUDE']:>10.3f}"
    )


# ============================================================
# 6. FRIEDMAN TEST
# ============================================================

n = len(EXPECTED_SCENARIOS)
k = len(GENERATORS)

rank_sums = {
    generator: 0.0
    for generator in GENERATORS
}

tie_sum = 0.0


for scenario_id in EXPECTED_SCENARIOS:

    values = [
        data[scenario_id][generator]
        for generator in GENERATORS
    ]

    ranks = average_ranks(
        values
    )

    for generator, rank in zip(
        GENERATORS,
        ranks
    ):
        rank_sums[
            generator
        ] += rank

    # Tie correction component
    counts = {}

    for value in values:
        counts[value] = (
            counts.get(value, 0)
            + 1
        )

    for count in counts.values():

        if count > 1:
            tie_sum += (
                count ** 3
                - count
            )


# Standard Friedman statistic

friedman_uncorrected = (
    12.0
    / (
        n
        * k
        * (k + 1)
    )
) * sum(
    rank_sums[g] ** 2
    for g in GENERATORS
) - (
    3
    * n
    * (k + 1)
)


# Tie correction

tie_correction = (
    1.0
    - (
        tie_sum
        / (
            n
            * k
            * (
                k ** 2 - 1
            )
        )
    )
)


if tie_correction <= 0:
    raise ValueError(
        "Invalid Friedman tie correction."
    )


friedman_stat = (
    friedman_uncorrected
    / tie_correction
)

df = k - 1

friedman_p = (
    chi_square_sf_df3(
        friedman_stat
    )
)


# ============================================================
# 7. KENDALL'S W
# ============================================================

kendalls_w = (
    friedman_stat
    / (
        n
        * (k - 1)
    )
)


print()
print("=" * 78)
print("B. FRIEDMAN OMNIBUS TEST")
print("=" * 78)

print()
print(
    f"Friedman chi-square: "
    f"{friedman_stat:.6f}"
)

print(
    f"Degrees of freedom:  "
    f"{df}"
)

print(
    f"p-value:             "
    f"{friedman_p:.8f}"
)

print(
    f"Kendall's W:         "
    f"{kendalls_w:.6f}"
)

print()
print("Mean ranks:")

for generator in GENERATORS:

    mean_rank = (
        rank_sums[generator]
        / n
    )

    print(
        f"  {generator}: "
        f"{mean_rank:.4f}"
    )


# ============================================================
# 8. WILCOXON SIGNED-RANK TEST
# ============================================================

def wilcoxon_signed_rank(
    x,
    y
):
    """
    Paired Wilcoxon signed-rank test.

    Zero differences are removed.

    Ties in absolute differences receive
    average ranks.

    Two-sided p-value uses normal
    approximation with tie correction
    and continuity correction.

    Also returns matched-pairs
    rank-biserial correlation:

        r_rb = (W+ - W-) /
               (W+ + W-)
    """

    differences = [
        a - b
        for a, b in zip(x, y)
    ]

    differences = [
        d
        for d in differences
        if abs(d) > 1e-12
    ]

    n_nonzero = len(
        differences
    )

    if n_nonzero == 0:

        return {
            "n": 0,
            "W_plus": 0.0,
            "W_minus": 0.0,
            "T": 0.0,
            "z": 0.0,
            "p": 1.0,
            "rank_biserial": 0.0,
        }


    abs_differences = [
        abs(d)
        for d in differences
    ]

    ranks = average_ranks(
        abs_differences
    )


    W_plus = sum(
        rank
        for difference, rank
        in zip(
            differences,
            ranks
        )
        if difference > 0
    )


    W_minus = sum(
        rank
        for difference, rank
        in zip(
            differences,
            ranks
        )
        if difference < 0
    )


    T = min(
        W_plus,
        W_minus
    )


    # --------------------------------------------------------
    # Expected value
    # --------------------------------------------------------

    mean_W = (
        n_nonzero
        * (
            n_nonzero + 1
        )
        / 4.0
    )


    # --------------------------------------------------------
    # Tie correction
    # --------------------------------------------------------

    tie_counts = {}

    for value in abs_differences:

        rounded_value = round(
            value,
            10
        )

        tie_counts[
            rounded_value
        ] = (
            tie_counts.get(
                rounded_value,
                0
            )
            + 1
        )


    tie_term = sum(
        count
        * (
            count + 1
        )
        * (
            2 * count + 1
        )
        for count
        in tie_counts.values()
        if count > 1
    )


    variance_W = (
        (
            n_nonzero
            * (
                n_nonzero + 1
            )
            * (
                2 * n_nonzero + 1
            )
        )
        - (
            tie_term / 2.0
        )
    ) / 24.0


    if variance_W <= 0:

        z = 0.0
        p_value = 1.0

    else:

        sd_W = math.sqrt(
            variance_W
        )

        # Continuity correction
        difference_from_mean = (
            abs(
                T - mean_W
            )
            - 0.5
        )

        if difference_from_mean < 0:
            difference_from_mean = 0

        z = (
            difference_from_mean
            / sd_W
        )

        p_value = (
            2.0
            * (
                1.0
                - normal_cdf(z)
            )
        )


    rank_total = (
        W_plus
        + W_minus
    )

    if rank_total == 0:

        rank_biserial = 0.0

    else:

        rank_biserial = (
            W_plus
            - W_minus
        ) / rank_total


    return {
        "n": n_nonzero,
        "W_plus": W_plus,
        "W_minus": W_minus,
        "T": T,
        "z": z,
        "p": p_value,
        "rank_biserial":
            rank_biserial,
    }


# ============================================================
# 9. ALL SIX PAIRWISE COMPARISONS
# ============================================================

pairwise_results = []


for generator_a, generator_b in (
    itertools.combinations(
        GENERATORS,
        2
    )
):

    scores_a = [
        data[scenario_id][
            generator_a
        ]
        for scenario_id
        in EXPECTED_SCENARIOS
    ]

    scores_b = [
        data[scenario_id][
            generator_b
        ]
        for scenario_id
        in EXPECTED_SCENARIOS
    ]

    result = (
        wilcoxon_signed_rank(
            scores_a,
            scores_b
        )
    )

    differences = [
        a - b
        for a, b
        in zip(
            scores_a,
            scores_b
        )
    ]

    mean_difference = (
        sum(differences)
        / len(differences)
    )

    median_difference = (
        sorted(differences)[
            len(differences) // 2 - 1
        ]
        + sorted(differences)[
            len(differences) // 2
        ]
    ) / 2.0

    pairwise_results.append({
        "generator_a":
            generator_a,

        "generator_b":
            generator_b,

        "n_nonzero":
            result["n"],

        "mean_difference_A_minus_B":
            mean_difference,

        "median_difference_A_minus_B":
            median_difference,

        "W_plus":
            result["W_plus"],

        "W_minus":
            result["W_minus"],

        "T":
            result["T"],

        "z":
            result["z"],

        "raw_p":
            result["p"],

        "rank_biserial":
            result[
                "rank_biserial"
            ],
    })


# ============================================================
# 10. HOLM CORRECTION
# ============================================================

m = len(
    pairwise_results
)

sorted_indices = sorted(
    range(m),
    key=lambda i:
        pairwise_results[i][
            "raw_p"
        ]
)


previous_adjusted = 0.0

for rank_position, index in enumerate(
    sorted_indices
):

    raw_p = pairwise_results[
        index
    ]["raw_p"]

    multiplier = (
        m - rank_position
    )

    adjusted = (
        raw_p
        * multiplier
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
        index
    ]["holm_p"] = adjusted

    pairwise_results[
        index
    ]["significant_holm_0.05"] = (
        adjusted < ALPHA
    )

    previous_adjusted = adjusted


# ============================================================
# 11. PRINT PAIRWISE RESULTS
# ============================================================

print()
print("=" * 78)
print("C. PAIRWISE WILCOXON SIGNED-RANK TESTS")
print("=" * 78)

print()
print(
    "Two-sided tests; Holm correction "
    "across six comparisons."
)

print()


for result in pairwise_results:

    print(
        f"{result['generator_a']} "
        f"vs "
        f"{result['generator_b']}"
    )

    print("-" * 45)

    print(
        f"Non-zero pairs:      "
        f"{result['n_nonzero']}"
    )

    print(
        f"Mean difference:     "
        f"{result['mean_difference_A_minus_B']:.4f}"
    )

    print(
        f"Median difference:   "
        f"{result['median_difference_A_minus_B']:.4f}"
    )

    print(
        f"W+:                  "
        f"{result['W_plus']:.4f}"
    )

    print(
        f"W-:                  "
        f"{result['W_minus']:.4f}"
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
        f"{result['rank_biserial']:.4f}"
    )

    print(
        f"Significant (Holm):  "
        f"{result['significant_holm_0.05']}"
    )

    print()


# ============================================================
# 12. SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_rows = []


# Friedman row

output_rows.append({
    "analysis":
        "Friedman",

    "comparison":
        "QWEN_LLAMA_GPT_CLAUDE",

    "statistic":
        round(
            friedman_stat,
            8
        ),

    "df":
        df,

    "raw_p":
        round(
            friedman_p,
            10
        ),

    "holm_p":
        "",

    "effect_size":
        round(
            kendalls_w,
            8
        ),

    "effect_size_type":
        "Kendalls_W",

    "significant_0.05":
        friedman_p < ALPHA,
})


# Pairwise rows

for result in pairwise_results:

    output_rows.append({
        "analysis":
            "Wilcoxon_signed_rank",

        "comparison":
            (
                f"{result['generator_a']}"
                f"_vs_"
                f"{result['generator_b']}"
            ),

        "statistic":
            round(
                result["T"],
                8
            ),

        "df":
            "",

        "raw_p":
            round(
                result["raw_p"],
                10
            ),

        "holm_p":
            round(
                result["holm_p"],
                10
            ),

        "effect_size":
            round(
                result[
                    "rank_biserial"
                ],
                8
            ),

        "effect_size_type":
            "matched_rank_biserial",

        "significant_0.05":
            result[
                "significant_holm_0.05"
            ],
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
            "comparison",
            "statistic",
            "df",
            "raw_p",
            "holm_p",
            "effect_size",
            "effect_size_type",
            "significant_0.05",
        ]
    )

    writer.writeheader()
    writer.writerows(
        output_rows
    )


# ============================================================
# 13. FINAL REPORT
# ============================================================

print("=" * 78)
print("STEP 4 COMPLETE")
print("=" * 78)

print()
print("Created:")
print(OUTPUT_FILE)

print()
print("Design:")
print(
    "  12 matched scenarios × "
    "4 generators"
)

print()
print("Primary omnibus test:")
print("  Friedman test")

print()
print("Post-hoc procedure:")
print(
    "  Pairwise Wilcoxon signed-rank "
    "+ Holm correction"
)

print()
print("Effect sizes:")
print("  Kendall's W")
print(
    "  Matched-pairs rank-biserial "
    "correlation"
)

print()
print("No API requests were made.")
print("No evaluation data were modified.")

print()
print(
    "Send the complete console output "
    "before interpreting statistical "
    "differences."
)

print("=" * 78)