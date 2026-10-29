import csv
import statistics
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

CONSENSUS_FILE = (
    OUTPUT_DIR
    / "consensus_sop_scores_48.csv"
)

MODEL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "generator_descriptive_statistics.csv"
)

JUDGES = [
    "GPT",
    "CLAUDE",
    "GEMINI"
]

GENERATORS = [
    "QWEN",
    "LLAMA",
    "GPT",
    "CLAUDE"
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
# 2. LOAD CONSOLIDATED 144-ROW DATASET
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
        data[anonymous_id] = {
            "metadata": {
                "generator": row["generator"],
                "scenario_id": row["scenario_id"],
                "process_id": row["process_id"],
                "variant": row["variant"],
                "original_sop_file": row["original_sop_file"],
                "sop_sha256": row["sop_sha256"],
            },
            "judges": {}
        }

    # Verify metadata consistency across judges
    metadata = data[anonymous_id]["metadata"]

    for field in [
        "generator",
        "scenario_id",
        "process_id",
        "variant",
        "original_sop_file",
        "sop_sha256",
    ]:
        if metadata[field] != row[field]:
            raise ValueError(
                f"Metadata mismatch for "
                f"{anonymous_id}: {field}"
            )

    if judge in data[anonymous_id]["judges"]:
        raise ValueError(
            f"Duplicate judge evaluation: "
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

    total_score = int(
        row["total_score"]
    )

    if total_score != sum(
        criterion_scores.values()
    ):
        raise ValueError(
            f"Total mismatch: "
            f"{anonymous_id} / {judge}"
        )

    data[anonymous_id]["judges"][judge] = {
        "total": total_score,
        "criteria": criterion_scores,
    }


# ============================================================
# 4. VERIFY COMPLETE DATA
# ============================================================

if set(data.keys()) != set(EXPECTED_IDS):
    raise ValueError(
        "Dataset does not contain exactly "
        "SOP_001 through SOP_048."
    )

for anonymous_id in EXPECTED_IDS:

    judge_data = data[
        anonymous_id
    ]["judges"]

    if set(judge_data.keys()) != set(JUDGES):
        raise ValueError(
            f"{anonymous_id}: expected all "
            f"three judges."
        )


# Verify 12 SOPs per generator

for generator in GENERATORS:

    count = sum(
        data[anonymous_id]["metadata"][
            "generator"
        ] == generator
        for anonymous_id in EXPECTED_IDS
    )

    if count != 12:
        raise ValueError(
            f"{generator}: expected 12 SOPs, "
            f"found {count}."
        )


print("=" * 76)
print("STEP 3 — CONSENSUS DATASET AND GENERATOR COMPARISON")
print("=" * 76)

print()
print("Dataset verified:")
print("  SOPs:                 48")
print("  Judges per SOP:        3")
print("  Evaluations:          144")
print("  Generator models:      4")
print("  SOPs per generator:   12")


# ============================================================
# 5. CREATE 48-ROW CONSENSUS DATASET
# ============================================================

consensus_rows = []


for anonymous_id in EXPECTED_IDS:

    item = data[anonymous_id]
    metadata = item["metadata"]
    judge_data = item["judges"]

    judge_totals = {
        judge: judge_data[judge]["total"]
        for judge in JUDGES
    }

    total_values = list(
        judge_totals.values()
    )

    consensus_total_mean = (
        statistics.mean(total_values)
    )

    consensus_total_median = (
        statistics.median(total_values)
    )

    total_min = min(total_values)
    total_max = max(total_values)

    total_range = (
        total_max - total_min
    )

    consensus_row = {
        "anonymous_id":
            anonymous_id,

        "generator":
            metadata["generator"],

        "scenario_id":
            metadata["scenario_id"],

        "process_id":
            metadata["process_id"],

        "variant":
            metadata["variant"],

        "GPT_total":
            judge_totals["GPT"],

        "CLAUDE_total":
            judge_totals["CLAUDE"],

        "GEMINI_total":
            judge_totals["GEMINI"],

        "consensus_total_mean":
            round(
                consensus_total_mean,
                6
            ),

        "consensus_total_median":
            round(
                consensus_total_median,
                6
            ),

        "judge_total_min":
            total_min,

        "judge_total_max":
            total_max,

        "judge_total_range":
            total_range,
    }

    # --------------------------------------------------------
    # Consensus criterion scores
    # --------------------------------------------------------

    for criterion in CRITERIA:

        criterion_values = [
            judge_data[judge][
                "criteria"
            ][criterion]
            for judge in JUDGES
        ]

        criterion_mean = (
            statistics.mean(
                criterion_values
            )
        )

        criterion_median = (
            statistics.median(
                criterion_values
            )
        )

        consensus_row[
            f"{criterion}_mean"
        ] = round(
            criterion_mean,
            6
        )

        consensus_row[
            f"{criterion}_median"
        ] = round(
            criterion_median,
            6
        )

    consensus_row[
        "original_sop_file"
    ] = metadata[
        "original_sop_file"
    ]

    consensus_row[
        "sop_sha256"
    ] = metadata[
        "sop_sha256"
    ]

    consensus_rows.append(
        consensus_row
    )


if len(consensus_rows) != 48:
    raise ValueError(
        f"Expected 48 consensus rows, "
        f"found {len(consensus_rows)}."
    )


# ============================================================
# 6. SAVE 48-ROW CONSENSUS DATASET
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

consensus_fields = [
    "anonymous_id",
    "generator",
    "scenario_id",
    "process_id",
    "variant",
    "GPT_total",
    "CLAUDE_total",
    "GEMINI_total",
    "consensus_total_mean",
    "consensus_total_median",
    "judge_total_min",
    "judge_total_max",
    "judge_total_range",
]

for criterion in CRITERIA:
    consensus_fields.append(
        f"{criterion}_mean"
    )
    consensus_fields.append(
        f"{criterion}_median"
    )

consensus_fields += [
    "original_sop_file",
    "sop_sha256",
]


with CONSENSUS_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=consensus_fields
    )

    writer.writeheader()
    writer.writerows(
        consensus_rows
    )


# ============================================================
# 7. DESCRIPTIVE GENERATOR COMPARISON
# ============================================================

print()
print("=" * 76)
print("A. CONSENSUS TOTAL SCORE BY GENERATOR")
print("=" * 76)


model_summary_rows = []


for generator in GENERATORS:

    generator_rows = [
        row
        for row in consensus_rows
        if row["generator"] == generator
    ]

    scores = [
        float(
            row["consensus_total_mean"]
        )
        for row in generator_rows
    ]

    n = len(scores)

    mean_score = statistics.mean(
        scores
    )

    median_score = statistics.median(
        scores
    )

    sd_score = statistics.stdev(
        scores
    )

    minimum = min(scores)
    maximum = max(scores)

    print()
    print(generator)
    print("-" * 40)

    print(f"N:       {n}")
    print(
        f"Mean:    {mean_score:.3f} / 40"
    )
    print(
        f"Median:  {median_score:.3f}"
    )
    print(
        f"SD:      {sd_score:.3f}"
    )
    print(
        f"Minimum: {minimum:.3f}"
    )
    print(
        f"Maximum: {maximum:.3f}"
    )

    model_summary_rows.append({
        "generator":
            generator,

        "n":
            n,

        "mean_consensus_total":
            round(
                mean_score,
                6
            ),

        "median_consensus_total":
            round(
                median_score,
                6
            ),

        "sd_consensus_total":
            round(
                sd_score,
                6
            ),

        "minimum_consensus_total":
            round(
                minimum,
                6
            ),

        "maximum_consensus_total":
            round(
                maximum,
                6
            ),
    })


# ============================================================
# 8. CONSENSUS CRITERION MEANS BY GENERATOR
# ============================================================

print()
print("=" * 76)
print("B. CONSENSUS CRITERION MEANS BY GENERATOR")
print("=" * 76)


for generator in GENERATORS:

    generator_rows = [
        row
        for row in consensus_rows
        if row["generator"] == generator
    ]

    print()
    print(generator)
    print("-" * 40)

    for criterion in CRITERIA:

        values = [
            float(
                row[
                    f"{criterion}_mean"
                ]
            )
            for row in generator_rows
        ]

        mean_value = statistics.mean(
            values
        )

        sd_value = statistics.stdev(
            values
        )

        print(
            f"{criterion}: "
            f"Mean={mean_value:.3f}, "
            f"SD={sd_value:.3f}"
        )


# ============================================================
# 9. JUDGE DISAGREEMENT WITHIN EACH GENERATOR
# ============================================================

print()
print("=" * 76)
print("C. JUDGE SCORE RANGE BY GENERATOR")
print("=" * 76)

print()
print(
    "Mean within-SOP difference between "
    "highest and lowest judge total:"
)


for generator in GENERATORS:

    generator_rows = [
        row
        for row in consensus_rows
        if row["generator"] == generator
    ]

    ranges = [
        int(
            row["judge_total_range"]
        )
        for row in generator_rows
    ]

    mean_range = statistics.mean(
        ranges
    )

    median_range = statistics.median(
        ranges
    )

    maximum_range = max(
        ranges
    )

    print()
    print(
        f"{generator}: "
        f"Mean range={mean_range:.3f}, "
        f"Median={median_range:.3f}, "
        f"Max={maximum_range}"
    )


# ============================================================
# 10. SAVE GENERATOR SUMMARY
# ============================================================

summary_fields = [
    "generator",
    "n",
    "mean_consensus_total",
    "median_consensus_total",
    "sd_consensus_total",
    "minimum_consensus_total",
    "maximum_consensus_total",
]


with MODEL_SUMMARY_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fields
    )

    writer.writeheader()
    writer.writerows(
        model_summary_rows
    )


# ============================================================
# 11. FINAL VERIFICATION
# ============================================================

# Verify each scenario occurs exactly four times
# in the 48-row consensus dataset.

scenario_counts = {}

for row in consensus_rows:

    scenario_id = row["scenario_id"]

    scenario_counts[
        scenario_id
    ] = (
        scenario_counts.get(
            scenario_id,
            0
        )
        + 1
    )


if len(scenario_counts) != 12:
    raise ValueError(
        f"Expected 12 scenarios, "
        f"found {len(scenario_counts)}."
    )


for scenario_id, count in scenario_counts.items():

    if count != 4:
        raise ValueError(
            f"{scenario_id}: expected "
            f"4 generator outputs, "
            f"found {count}."
        )


# Verify A/B balance per generator

for generator in GENERATORS:

    generator_rows = [
        row
        for row in consensus_rows
        if row["generator"] == generator
    ]

    variant_counts = {
        "A": 0,
        "B": 0
    }

    for row in generator_rows:

        variant = row["variant"]

        if variant not in variant_counts:
            raise ValueError(
                f"Unexpected variant: {variant}"
            )

        variant_counts[variant] += 1

    if variant_counts["A"] != 6:
        raise ValueError(
            f"{generator}: expected 6 A scenarios, "
            f"found {variant_counts['A']}."
        )

    if variant_counts["B"] != 6:
        raise ValueError(
            f"{generator}: expected 6 B scenarios, "
            f"found {variant_counts['B']}."
        )


# ============================================================
# 12. FINAL REPORT
# ============================================================

print()
print("=" * 76)
print("STEP 3 COMPLETE")
print("=" * 76)

print()
print("Created:")
print(CONSENSUS_FILE)
print(MODEL_SUMMARY_FILE)

print()
print("Verification:")
print("  Consensus SOP rows:       48/48")
print("  Generators:                4")
print("  SOPs per generator:       12")
print("  Scenarios:                12")
print("  Outputs per scenario:      4")
print("  A/B per generator:       6/6")

print()
print("No API requests were made.")
print("No raw evaluation results were modified.")

print()
print(
    "Send the complete console output "
    "before proceeding to statistical "
    "generator comparisons."
)

print("=" * 76)