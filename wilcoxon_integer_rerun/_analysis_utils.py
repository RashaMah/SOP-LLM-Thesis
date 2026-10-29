from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "FINAL_EVALUATION_RESULTS" / "CONSOLIDATED" / "evaluation_summary_144.csv"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

CRITERIA = [
    "C1_completeness","C2_correct_sequence","C3_role_clarity",
    "C4_input_output_clarity","C5_exception_handling",
    "C6_constraint_adherence","C7_groundedness","C8_usability",
]
GENERATORS = ["QWEN","LLAMA","GPT","CLAUDE"]
JUDGES = ["GPT","CLAUDE","GEMINI"]
PAIRS = [
    ("QWEN","LLAMA"),("QWEN","GPT"),("QWEN","CLAUDE"),
    ("LLAMA","GPT"),("LLAMA","CLAUDE"),("GPT","CLAUDE")
]

def load_summary():
    df = pd.read_csv(SUMMARY)
    # The archived CSV stores criteria as e.g. "C1_completeness_score";
    # accept both naming conventions.
    df = df.rename(columns={f"{c}_score": c for c in CRITERIA if f"{c}_score" in df.columns})
    required = {"anonymous_id","generator","scenario_id","judge","total_score",*CRITERIA}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != 144:
        raise ValueError(f"Expected 144 rows, found {len(df)}")
    return df

def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted

def rank_biserial_paired(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0, 0
    ranks = rankdata(np.abs(d), method="average")
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    denom = w_plus + w_minus
    return float((w_plus - w_minus) / denom), int(len(d))

def consensus_sop_scores(df):
    """Consensus = mean of the three judges (reported in the thesis).

    Columns ending in "_sum" hold the integer sum of the three judges.
    The sum is rank-equivalent to the mean, so the tests give the same
    result, but it avoids floating-point noise breaking ties such as
    1/12 - 1/12 (all hypothesis tests therefore use the "_sum" columns).
    """
    agg = (
        df.groupby(["anonymous_id","generator","scenario_id"], as_index=False)
          .agg(total_score=("total_score","mean"),
               total_sum=("total_score","sum"),
               **{c: (c,"mean") for c in CRITERIA},
               **{f"{c}_sum": (c,"sum") for c in CRITERIA})
    )
    if len(agg) != 48:
        raise ValueError(f"Expected 48 consensus SOP rows, found {len(agg)}")
    return agg

def process_family(s):
    return str(s).split("-")[0]

def variant(s):
    return str(s).split("-")[-1]
