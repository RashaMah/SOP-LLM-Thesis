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
    / "ANYSIS_FIGURES"
)

PNG_FILE = OUTPUT_DIR / "Figure_5_1_Generator_Scores.png"
PDF_FILE = OUTPUT_DIR / "Figure_5_1_Generator_Scores.pdf"


# ============================================================
# 2. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("FIGURE 5.1 — CONSENSUS SOP SCORES BY GENERATOR")
print("=" * 70)

print("\nColumns found:")
print(list(df.columns))


# ============================================================
# 3. DETECT REQUIRED COLUMNS
# ============================================================

def find_column(options):
    for option in options:
        if option in df.columns:
            return option

    raise ValueError(
        "Could not find any of these columns: "
        + ", ".join(options)
    )


GENERATOR_COL = find_column(
    ["generator", "generator_model"]
)

SCORE_COL = find_column(
    [
        "consensus_total_mean",
        "consensus_total",
        "mean_total_score",
        "total_score_mean",
    ]
)

print("\nDetected:")
print(f"Generator column: {GENERATOR_COL}")
print(f"Score column:     {SCORE_COL}")


# ============================================================
# 4. NORMALIZE GENERATOR NAMES
# ============================================================

def normalize_generator(value):
    value = str(value).strip().upper()

    if "QWEN" in value:
        return "Qwen3.5-27B"

    if "LLAMA" in value:
        return "Llama 4 Maverick"

    if "GPT" in value:
        return "GPT-5.6 Sol"

    if "CLAUDE" in value:
        return "Claude Sonnet 5"

    return value


df["Generator_Display"] = df[GENERATOR_COL].apply(
    normalize_generator
)

generator_order = [
    "Qwen3.5-27B",
    "Llama 4 Maverick",
    "GPT-5.6 Sol",
    "Claude Sonnet 5",
]


# ============================================================
# 5. VERIFY DATA
# ============================================================

if len(df) != 48:
    raise ValueError(
        f"Expected 48 SOPs, found {len(df)}."
    )

counts = (
    df["Generator_Display"]
    .value_counts()
)

print("\nSOPs per generator:")

for generator in generator_order:
    count = int(counts.get(generator, 0))
    print(f"  {generator}: {count}")

    if count != 12:
        raise ValueError(
            f"Expected 12 SOPs for {generator}, "
            f"found {count}."
        )


# ============================================================
# 6. PREPARE DATA
# ============================================================

plot_data = [
    df.loc[
        df["Generator_Display"] == generator,
        SCORE_COL
    ].astype(float).values
    for generator in generator_order
]


# ============================================================
# 7. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

positions = list(
    range(1, len(generator_order) + 1)
)

ax.boxplot(
    plot_data,
    positions=positions,
    widths=0.55,
    showfliers=False,
    patch_artist=False,
)

# Add individual observations.
# Fixed offsets are used instead of random jitter so that the
# figure is exactly reproducible every time the script is run.

offsets = [
    -0.18,
    -0.15,
    -0.12,
    -0.09,
    -0.06,
    -0.03,
    0.03,
    0.06,
    0.09,
    0.12,
    0.15,
    0.18,
]

for position, values in zip(
    positions,
    plot_data
):
    sorted_values = sorted(values)

    x_values = [
        position + offset
        for offset in offsets
    ]

    ax.scatter(
        x_values,
        sorted_values,
        s=28,
        alpha=0.75,
        zorder=3,
    )


# ============================================================
# 8. FORMATTING
# ============================================================

ax.set_xticks(positions)

ax.set_xticklabels(
    generator_order,
    fontsize=10,
)

ax.set_ylabel(
    "Consensus total score (maximum = 40)",
    fontsize=11,
)

ax.set_xlabel(
    "Generator",
    fontsize=11,
)

ax.set_ylim(
    30,
    40.5,
)

ax.set_yticks(
    range(30, 41, 2)
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()


# ============================================================
# 9. SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# 10. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FIGURE 5.1 COMPLETE")
print("=" * 70)

print("\nCreated:")
print(PNG_FILE)
print(PDF_FILE)

print("\nFigure contents:")
print("  4 generators")
print("  12 SOP observations per generator")
print("  Boxplots show score distributions")
print("  Points show all individual SOP consensus scores")

print("\nNo source data were modified.")
print("=" * 70)