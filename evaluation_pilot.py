import csv
import statistics
from collections import Counter
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "judge_descriptive_statistics.csv"
)

JUDGES = [
    "GPT",
    "CLAUDE",
    "GEMINI"
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
        f"Expected 144 evaluation rows, "
        f"found {len(rows)}."
    )


# ============================================================
# 3. VALIDATE DATA
# ============================================================

for row in rows:

    judge = row["judge"]

    if judge not in JUDGES:
        raise ValueError(
            f"Unexpected judge: {judge}"
        )

    total = int(row["total_score"])

    if not 8 <= total <= 40:
        raise ValueError(
            f"Invalid total score: {total}"
        )

    for criterion in CRITERIA:

        column = f"{criterion}_score"

        score = int(row[column])

        if score not in [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Invalid score in {column}: {score}"
            )


# Verify 48 evaluations per judge

for judge in JUDGES:

    judge_rows = [
        row for row in rows
        if row["judge"] == judge
    ]

    if len(judge_rows) != 48:
        raise ValueError(
            f"{judge}: expected 48 evaluations, "
            f"found {len(judge_rows)}."
        )


print("=" * 72)
print("STEP 1 — DESCRIPTIVE JUDGE ANALYSIS")
print("=" * 72)

print()
print("Dataset verified:")
print("  Total evaluations: 144")
print("  GPT:                48")
print("  CLAUDE:             48")
print("  GEMINI:             48")


# ============================================================
# 4. TOTAL SCORE STATISTICS
# ============================================================

print()
print("=" * 72)
print("A. TOTAL SCORE BEHAVIOR")
print("=" * 72)

descriptive_rows = []


for judge in JUDGES:

    judge_rows = [
        row for row in rows
        if row["judge"] == judge
    ]

    totals = [
        int(row["total_score"])
        for row in judge_rows
    ]

    n = len(totals)

    mean_score = statistics.mean(totals)
    median_score = statistics.median(totals)

    sd_score = statistics.stdev(totals)

    minimum = min(totals)
    maximum = max(totals)

    perfect_count = sum(
        score == 40
        for score in totals
    )

    perfect_percent = (
        perfect_count / n
    ) * 100


    print()
    print(judge)
    print("-" * 40)
    print(f"N:             {n}")
    print(f"Mean:          {mean_score:.3f} / 40")
    print(f"Median:        {median_score:.3f}")
    print(f"SD:            {sd_score:.3f}")
    print(f"Minimum:       {minimum}")
    print(f"Maximum:       {maximum}")
    print(
        f"40/40 scores:  "
        f"{perfect_count}/{n} "
        f"({perfect_percent:.1f}%)"
    )


    descriptive_rows.append({
        "judge": judge,
        "n": n,
        "mean_total": round(
            mean_score,
            4
        ),
        "median_total": round(
            median_score,
            4
        ),
        "sd_total": round(
            sd_score,
            4
        ),
        "minimum_total": minimum,
        "maximum_total": maximum,
        "perfect_40_count": perfect_count,
        "perfect_40_percent": round(
            perfect_percent,
            2
        ),
    })


# ============================================================
# 5. TOTAL SCORE FREQUENCY DISTRIBUTION
# ============================================================

print()
print("=" * 72)
print("B. TOTAL SCORE FREQUENCY DISTRIBUTION")
print("=" * 72)


for judge in JUDGES:

    judge_rows = [
        row for row in rows
        if row["judge"] == judge
    ]

    totals = [
        int(row["total_score"])
        for row in judge_rows
    ]

    frequencies = Counter(totals)

    print()
    print(judge)
    print("-" * 40)

    for score in sorted(
        frequencies.keys()
    ):
        count = frequencies[score]
        percent = (
            count / len(totals)
        ) * 100

        print(
            f"{score:2d}/40: "
            f"{count:2d} "
            f"({percent:5.1f}%)"
        )


# ============================================================
# 6. CRITERION-LEVEL DESCRIPTIVE STATISTICS
# ============================================================

print()
print("=" * 72)
print("C. MEAN SCORE BY CRITERION")
print("=" * 72)


for judge in JUDGES:

    print()
    print(judge)
    print("-" * 40)

    judge_rows = [
        row for row in rows
        if row["judge"] == judge
    ]

    for criterion in CRITERIA:

        column = f"{criterion}_score"

        scores = [
            int(row[column])
            for row in judge_rows
        ]

        mean_score = statistics.mean(
            scores
        )

        sd_score = statistics.stdev(
            scores
        )

        print(
            f"{criterion}: "
            f"Mean={mean_score:.3f}, "
            f"SD={sd_score:.3f}"
        )


# ============================================================
# 7. CRITERION SCORE DISTRIBUTIONS
# ============================================================

print()
print("=" * 72)
print("D. CRITERION SCORE DISTRIBUTIONS (1–5)")
print("=" * 72)


for judge in JUDGES:

    print()
    print("#" * 72)
    print(judge)
    print("#" * 72)

    judge_rows = [
        row for row in rows
        if row["judge"] == judge
    ]

    for criterion in CRITERIA:

        column = f"{criterion}_score"

        scores = [
            int(row[column])
            for row in judge_rows
        ]

        frequencies = Counter(scores)

        print()
        print(criterion)

        for score in range(1, 6):

            count = frequencies.get(
                score,
                0
            )

            percent = (
                count / len(scores)
            ) * 100

            print(
                f"  Score {score}: "
                f"{count:2d}/48 "
                f"({percent:5.1f}%)"
            )


# ============================================================
# 8. CRITERION CEILING ANALYSIS
# ============================================================

print()
print("=" * 72)
print("E. CRITERION CEILING ANALYSIS")
print("=" * 72)

print()
print(
    "Percentage of criterion ratings "
    "that received 5/5:"
)


for judge in JUDGES:

    judge_rows = [
        row for row in rows
        if row["judge"] == judge
    ]

    all_scores = []

    for row in judge_rows:

        for criterion in CRITERIA:

            all_scores.append(
                int(
                    row[
                        f"{criterion}_score"
                    ]
                )
            )

    five_count = sum(
        score == 5
        for score in all_scores
    )

    total_ratings = len(all_scores)

    five_percent = (
        five_count / total_ratings
    ) * 100

    print(
        f"{judge}: "
        f"{five_count}/{total_ratings} "
        f"({five_percent:.1f}%)"
    )


# ============================================================
# 9. SAVE DESCRIPTIVE SUMMARY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fieldnames = [
    "judge",
    "n",
    "mean_total",
    "median_total",
    "sd_total",
    "minimum_total",
    "maximum_total",
    "perfect_40_count",
    "perfect_40_percent",
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
        descriptive_rows
    )


# ============================================================
# 10. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("STEP 1 COMPLETE")
print("=" * 72)

print()
print("Created:")
print(OUTPUT_FILE)

print()
print("No API requests were made.")
print("No evaluation results were modified.")

print()
print(
    "Next analysis step: "
    "inter-judge agreement."
)

print("=" * 72)

#Judge behavior differs substantially. GPT is the strictest and most discriminative judge (mean 36.02/40, SD 2.89, only 4.2% perfect scores). Claude is more lenient (mean 38.00, SD 1.87, 27.1% perfect). Gemini is extremely lenient and shows a strong ceiling effect (mean 39.67, SD 1.02, 87.5% perfect SOP totals and 96.9% of all criterion ratings equal 5).
#Especially important: Gemini assigned 5/5 to every single SOP for C4 and C5. Consequently, Gemini cannot discriminate between SOPs at all on those two dimensions. GPT, by contrast, shows much greater differentiation, particularly for C7 Groundedness (mean 3.625) and C5 Exception Handling (mean 4.312).
#This does not mean Gemini's evaluations are “wrong,” and we should absolutely not rerun or alter them. It means the judges apply the rubric differently, which is precisely why using three judges and retaining their individual ratings was useful.