"""
CORRECTION RE-RUN (written after the original analysis).
The original scripts in the repository root ran the signed-rank tests on
consensus means rounded to six decimals; rounding noise can break ties
(e.g. 1/12 vs -1/12). This script repeats the tests on integer sums of the
three judge totals, which are rank-equivalent to the means. The thesis
reports the values produced here.
Consensus scores; RQ2 Friedman + six paired Wilcoxon tests; Holm correction.
"""
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from _analysis_utils import OUT, load_summary, consensus_sop_scores, GENERATORS, PAIRS, holm_adjust, rank_biserial_paired

df = load_summary()
cons = consensus_sop_scores(df)
out = cons.copy()
out.to_csv(OUT / "consensus_sop_scores_48.csv", index=False)

print("\nGENERATOR DESCRIPTIVES")
for g in GENERATORS:
    x = cons.loc[cons.generator==g, "total_score"]
    print(g, "N",len(x),"mean",x.mean(),"median",x.median(),"sd",x.std(ddof=1),"min",x.min(),"max",x.max())

wide = cons.pivot(index="scenario_id", columns="generator", values="total_sum")[GENERATORS]  # integer sums, see _analysis_utils
stat, p = friedmanchisquare(*(wide[g] for g in GENERATORS))
print("\nFRIEDMAN RQ2", stat, p, "Kendall_W", stat/(len(wide)*(len(GENERATORS)-1)))

rows = []
for a,b in PAIRS:
    x,y = wide[a].to_numpy(float), wide[b].to_numpy(float)
    w = wilcoxon(x,y,alternative="two-sided",zero_method="wilcox",method="auto")
    rrb, n_nonzero = rank_biserial_paired(x,y)
    rows.append({"comparison":f"{a} vs {b}","T":float(w.statistic),"raw_p":float(w.pvalue),
                 "r_rb":rrb,"n_pairs_total":len(x),"n_nonzero_pairs":n_nonzero})
adj = holm_adjust([r["raw_p"] for r in rows])
for r,a in zip(rows,adj):
    r["holm_p"] = float(a)
    print(r)
pd.DataFrame(rows).to_csv(OUT / "rq2_pairwise_wilcoxon.csv", index=False)
