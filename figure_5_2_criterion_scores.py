from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
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
    / "ANALYSIS_FIGURES"
)

PNG_FILE = OUTPUT_DIR / "Figure_5_2_Criterion_Scores.png"
PDF_FILE = OUTPUT_DIR / "Figure_5_2_Criterion_Scores.pdf"


# ============================================================
# 2. CONFIGURATION
# ============================================================

GENERATOR_ORDER = [
    "QWEN",
    "LLAMA",
    "GPT",
    "CLAUDE",
]

GENERATOR_LABELS = {
    "QWEN": "Qwen3.5-27B",
    "LLAMA": "Llama 4 Maverick",
    "GPT": "GPT-5.6 Sol",
    "CLAUDE": "Claude Sonnet 5",
}

CRITERIA = {
    "C1": "C1_completeness_mean",
    "C2": "C2_correct_sequence_mean",
    "C3": "C3_role_clarity_mean",
    "C4": "C4_input_output_clarity_mean",
    "C5": "C5_exception_handling_mean",
    "C6": "C6_constraint_adherence_mean",
    "C7": "C7_groundedness_mean",
    "C8": "C8_usability_mean",
}


# ============================================================
# 3. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 72)
print("FIGURE 5.2 — CRITERION-LEVEL CONSENSUS SCORES")
print("=" * 72)

print(f"\nLoaded SOPs: {len(df)}")

if len(df) != 48:
    raise ValueError(
        f"Expected 48 SOPs, found {len(df)}."
    )


# ============================================================
# 4. VERIFY COLUMNS
# ============================================================

required_columns = [
    "generator",
    *CRITERIA.values(),
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print("\nAll required criterion columns found.")


# ============================================================
# 5. NORMALIZE GENERATOR NAMES
# ============================================================

def normalize_generator(value):
    value = str(value).strip().upper()

    if "QWEN" in value:
        return "QWEN"

    if "LLAMA" in value:
        return "LLAMA"

    if "GPT" in value:
        return "GPT"

    if "CLAUDE" in value:
        return "CLAUDE"

    return value


df["generator_normalized"] = (
    df["generator"]
    .apply(normalize_generator)
)


# ============================================================
# 6. VERIFY 12 SOPs PER GENERATOR
# ============================================================

print("\nSOPs per generator:")

for generator in GENERATOR_ORDER:

    count = int(
        (
            df["generator_normalized"]
            == generator
        ).sum()
    )

    print(
        f"  {GENERATOR_LABELS[generator]}: "
        f"{count}"
    )

    if count != 12:
        raise ValueError(
            f"Expected 12 SOPs for {generator}, "
            f"found {count}."
        )


# ============================================================
# 7. CALCULATE MEANS
# ============================================================

criterion_means = {}

for generator in GENERATOR_ORDER:

    generator_df = df[
        df["generator_normalized"]
        == generator
    ]

    criterion_means[generator] = [
        generator_df[column].mean()
        for column in CRITERIA.values()
    ]


print("\nCriterion means:")

for generator in GENERATOR_ORDER:

    values = criterion_means[generator]

    formatted = ", ".join(
        f"{criterion}={value:.3f}"
        for criterion, value in zip(
            CRITERIA.keys(),
            values
        )
    )

    print(
        f"  {GENERATOR_LABELS[generator]}: "
        f"{formatted}"
    )


# ============================================================
# 8. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.5, 5.8)
)

x = list(
    range(1, len(CRITERIA) + 1)
)

markers = [
    "o",
    "s",
    "^",
    "D",
]

linestyles = [
    "-",
    "--",
    "-.",
    ":",
]

for generator, marker, linestyle in zip(
    GENERATOR_ORDER,
    markers,
    linestyles,
):

    ax.plot(
        x,
        criterion_means[generator],
        marker=marker,
        linestyle=linestyle,
        linewidth=1.5,
        markersize=6,
        label=GENERATOR_LABELS[generator],
    )


# ============================================================
# 9. FORMATTING
# ============================================================

ax.set_xticks(x)

ax.set_xticklabels(
    list(CRITERIA.keys()),
    fontsize=10,
)

ax.set_xlabel(
    "Evaluation criterion",
    fontsize=11,
)

ax.set_ylabel(
    "Mean consensus criterion score (maximum = 5)",
    fontsize=11,
)

# Scores are concentrated near the top of the scale.
# The restricted axis improves readability while the label
# explicitly states that the full criterion maximum is 5.

ax.set_ylim(
    3.8,
    5.05,
)

ax.set_yticks(
    [
        3.8,
        4.0,
        4.2,
        4.4,
        4.6,
        4.8,
        5.0,
    ]
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
    ncol=2,
    fontsize=9,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()


# ============================================================
# 10. SAVE FIGURE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    PNG_FILE,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_FILE,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 11. COMPLETE
# ============================================================

print("\n" + "=" * 72)
print("FIGURE 5.2 COMPLETE")
print("=" * 72)

print("\nCreated:")
print(PNG_FILE)
print(PDF_FILE)

print("\nFigure contents:")
print("  8 evaluation criteria")
print("  4 generation models")
print("  Means calculated from 12 SOPs per generator")
print("  Each SOP criterion value is the three-judge consensus mean")

print("\nCriterion labels:")
print("  C1 = Completeness")
print("  C2 = Correct sequence")
print("  C3 = Role clarity")
print("  C4 = Input/output clarity")
print("  C5 = Exception handling")
print("  C6 = Constraint adherence")
print("  C7 = Groundedness / no hallucination")
print("  C8 = Usability")

print("\nNo source data were modified.")
print("=" * 72)