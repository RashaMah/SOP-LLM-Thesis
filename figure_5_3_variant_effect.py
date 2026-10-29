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

PNG_FILE = OUTPUT_DIR / "Figure_5_3_Variant_A_vs_B.png"
PDF_FILE = OUTPUT_DIR / "Figure_5_3_Variant_A_vs_B.pdf"


# ============================================================
# 2. CONFIGURATION
# ============================================================

PROCESS_ORDER = [
    "GR",
    "RL",
    "IR",
    "OP",
    "ST",
    "PD",
]

PROCESS_LABELS = {
    "GR": "Goods\nReceipt",
    "RL": "Reverse\nLogistics",
    "IR": "Inventory\nReplenishment",
    "OP": "Order\nPicking",
    "ST": "Stock\nTransfer",
    "PD": "Packing &\nDispatch",
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
print("FIGURE 5.3 — VARIANT A vs VARIANT B")
print("=" * 72)

print(f"\nLoaded SOPs: {len(df)}")

if len(df) != 48:
    raise ValueError(
        f"Expected 48 SOPs, found {len(df)}."
    )


# ============================================================
# 4. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "process_id",
    "variant",
    "consensus_total_mean",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print("\nRequired columns verified.")


# ============================================================
# 5. NORMALIZE VALUES
# ============================================================

df["process_normalized"] = (
    df["process_id"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["variant_normalized"] = (
    df["variant"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# 6. VERIFY DESIGN
# ============================================================

for process in PROCESS_ORDER:

    for variant in ["A", "B"]:

        subset = df[
            (df["process_normalized"] == process)
            &
            (df["variant_normalized"] == variant)
        ]

        if len(subset) != 4:
            raise ValueError(
                f"Expected 4 generator observations for "
                f"{process}-{variant}, found {len(subset)}."
            )

print(
    "\nDesign verified: "
    "4 generator observations per process × variant."
)


# ============================================================
# 7. CALCULATE PROCESS-LEVEL MEANS
# ============================================================

variant_a = []
variant_b = []
differences = []

print("\nProcess-level results:")
print(
    f"{'Process':<10}"
    f"{'A mean':>12}"
    f"{'B mean':>12}"
    f"{'B - A':>12}"
)

print("-" * 46)

for process in PROCESS_ORDER:

    a_values = df.loc[
        (df["process_normalized"] == process)
        &
        (df["variant_normalized"] == "A"),
        "consensus_total_mean",
    ].astype(float)

    b_values = df.loc[
        (df["process_normalized"] == process)
        &
        (df["variant_normalized"] == "B"),
        "consensus_total_mean",
    ].astype(float)

    a_mean = a_values.mean()
    b_mean = b_values.mean()

    difference = b_mean - a_mean

    variant_a.append(a_mean)
    variant_b.append(b_mean)
    differences.append(difference)

    print(
        f"{process:<10}"
        f"{a_mean:>12.3f}"
        f"{b_mean:>12.3f}"
        f"{difference:>+12.3f}"
    )


# ============================================================
# 8. OVERALL CHECK
# ============================================================

overall_a = sum(variant_a) / len(variant_a)
overall_b = sum(variant_b) / len(variant_b)

print("\nOverall process-level means:")
print(f"  Variant A: {overall_a:.3f}")
print(f"  Variant B: {overall_b:.3f}")
print(f"  B - A:     {overall_b - overall_a:+.3f}")


# ============================================================
# 9. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.5, 5.8)
)

x = list(
    range(1, len(PROCESS_ORDER) + 1)
)

# Plot A and B and connect the matched process pair.
for position, a_score, b_score in zip(
    x,
    variant_a,
    variant_b,
):
    ax.plot(
        [position, position],
        [a_score, b_score],
        linewidth=1.0,
        alpha=0.5,
        zorder=1,
    )

ax.scatter(
    x,
    variant_a,
    marker="o",
    s=65,
    label="Variant A — standard",
    zorder=3,
)

ax.scatter(
    x,
    variant_b,
    marker="s",
    s=65,
    label="Variant B — exception/constraint-rich",
    zorder=3,
)


# ============================================================
# 10. FORMATTING
# ============================================================

ax.axhline(
    overall_a,
    linestyle="--",
    linewidth=1.0,
    alpha=0.45,
)

ax.axhline(
    overall_b,
    linestyle=":",
    linewidth=1.0,
    alpha=0.45,
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        PROCESS_LABELS[p]
        for p in PROCESS_ORDER
    ],
    fontsize=9,
)

ax.set_xlabel(
    "Logistics process family",
    fontsize=11,
)

ax.set_ylabel(
    "Mean consensus total score (maximum = 40)",
    fontsize=11,
)

ax.set_ylim(
    35.5,
    40.0,
)

ax.set_yticks(
    [
        35.5,
        36.0,
        36.5,
        37.0,
        37.5,
        38.0,
        38.5,
        39.0,
        39.5,
        40.0,
    ]
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
    fontsize=9,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()


# ============================================================
# 11. SAVE
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
# 12. COMPLETE
# ============================================================

print("\n" + "=" * 72)
print("FIGURE 5.3 COMPLETE")
print("=" * 72)

print("\nCreated:")
print(PNG_FILE)
print(PDF_FILE)

print("\nExpected reference result:")
print("  Overall Variant A ≈ 37.875")
print("  Overall Variant B ≈ 37.917")
print("  Overall B - A     ≈ +0.042")

print("\nNo statistical test is recomputed here.")
print("This figure visualizes the already completed RQ3 analysis.")
print("No source data were modified.")

print("=" * 72)