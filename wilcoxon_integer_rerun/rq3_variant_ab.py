"""
CORRECTION RE-RUN (written after the original analysis).
The original scripts in the repository root ran the signed-rank tests on
consensus means rounded to six decimals; rounding noise can break ties
(e.g. 1/12 vs -1/12). This script repeats the tests on integer sums of the
three judge totals, which are rank-equivalent to the means. The thesis
reports the values produced here.
RQ3 matched A/B tests using the consensus SOP total scores.
Reports total matched pairs and non-zero pairs actually used by Wilcoxon.
"""
import pandas as pd
from scipy.stats import wilcoxon
from _analysis_utils import OUT, load_summary, consensus_sop_scores, GENERATORS, holm_adjust, rank_biserial_paired, process_family, variant

cons = consensus_sop_scores(load_summary())
cons["process_family"] = cons["scenario_id"].map(process_family)
cons["variant"] = cons["scenario_id"].map(variant)

rows = []
for g in GENERATORS:
    sub = cons[cons.generator==g].pivot(index="process_family",columns="variant",values="total_sum")
    x,y = sub["A"].to_numpy(float), sub["B"].to_numpy(float)
    w = wilcoxon(x,y,alternative="two-sided",zero_method="wilcox",method="auto")
    rrb,nz = rank_biserial_paired(y,x)  # positive = B > A
    rows.append({"generator":g,"A_mean":x.mean()/3,"B_mean":y.mean()/3,"B_minus_A":(y-x).mean()/3,
                 "T":float(w.statistic),"raw_p":float(w.pvalue),"r_rb_B_minus_A":rrb,
                 "n_pairs_total":len(x),"n_nonzero_pairs":nz})

adj = holm_adjust([r["raw_p"] for r in rows])
for r,a in zip(rows,adj):
    r["holm_p"] = float(a)
    print(r)
pd.DataFrame(rows).to_csv(OUT / "rq3_per_generator_ab.csv", index=False)

# Overall: first average the four generators within each process-family x variant,
# then compare the six A/B process-family pairs.
proc = (cons.groupby(["process_family","variant"],as_index=False)["total_sum"].sum()
           .pivot(index="process_family",columns="variant",values="total_sum"))  # integer: 4 generators x 3 judges
x,y = proc["A"].to_numpy(float), proc["B"].to_numpy(float)
w = wilcoxon(x,y,alternative="two-sided",zero_method="wilcox",method="auto")
rrb,nz = rank_biserial_paired(y,x)
print("\nOVERALL PROCESS-PAIRED A/B")
print({"A_mean":x.mean()/12,"B_mean":y.mean()/12,"B_minus_A":(y-x).mean()/12,
       "T":float(w.statistic),"p":float(w.pvalue),
       "r_rb_B_minus_A":rrb,"n_pairs_total":len(x),"n_nonzero_pairs":nz})
