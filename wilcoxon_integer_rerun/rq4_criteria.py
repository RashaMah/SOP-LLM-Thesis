"""
CORRECTION RE-RUN (written after the original analysis).
The original scripts in the repository root ran the signed-rank tests on
consensus means rounded to six decimals; rounding noise can break ties
(e.g. 1/12 vs -1/12). This script repeats the tests on integer sums of the
three judge totals, which are rank-equivalent to the means. The thesis
reports the values produced here.
RQ4 criterion-level Friedman tests, Holm across 8 omnibus tests,
then six pairwise Wilcoxon tests within each criterion that survives Holm.
"""
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from _analysis_utils import OUT, load_summary, consensus_sop_scores, CRITERIA, GENERATORS, PAIRS, holm_adjust, rank_biserial_paired

cons = consensus_sop_scores(load_summary())
omnibus = []

for c in CRITERIA:
    wide = cons.pivot(index="scenario_id",columns="generator",values=f"{c}_sum")[GENERATORS]  # integer sums
    stat,p = friedmanchisquare(*(wide[g] for g in GENERATORS))
    omnibus.append({"criterion":c,"chi2":float(stat),"raw_p":float(p),
                    "kendall_W":float(stat/(len(wide)*(len(GENERATORS)-1)))})
adj = holm_adjust([r["raw_p"] for r in omnibus])
for r,a in zip(omnibus,adj):
    r["holm_p"] = float(a)
pd.DataFrame(omnibus).to_csv(OUT / "rq4_criterion_friedman.csv", index=False)

posthoc = []
for omni in omnibus:
    if omni["holm_p"] >= 0.05:
        continue
    c = omni["criterion"]
    wide = cons.pivot(index="scenario_id",columns="generator",values=f"{c}_sum")[GENERATORS]  # integer sums
    local = []
    for a,b in PAIRS:
        x,y = wide[a].to_numpy(float),wide[b].to_numpy(float)
        if (x==y).all():
            T,p = 0.0,1.0
        else:
            w = wilcoxon(x,y,alternative="two-sided",zero_method="wilcox",method="auto")
            T,p = float(w.statistic),float(w.pvalue)
        rrb,nz = rank_biserial_paired(x,y)
        local.append({"criterion":c,"comparison":f"{a} vs {b}","T":T,"raw_p":p,
                      "r_rb":rrb,"n_pairs_total":len(x),"n_nonzero_pairs":nz})
    ladj = holm_adjust([r["raw_p"] for r in local])
    for r,a in zip(local,ladj):
        r["holm_p_within_criterion"] = float(a)
        posthoc.append(r)

print(pd.DataFrame(omnibus).to_string(index=False))
print("\nPOSTHOC")
print(pd.DataFrame(posthoc).to_string(index=False))
pd.DataFrame(posthoc).to_csv(OUT / "rq4_criterion_posthoc.csv", index=False)
