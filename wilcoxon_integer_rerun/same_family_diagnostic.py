"""
CORRECTION RE-RUN (written after the original analysis).
The original scripts in the repository root ran the signed-rank tests on
consensus means rounded to six decimals; rounding noise can break ties
(e.g. 1/12 vs -1/12). This script repeats the tests on integer sums of the
three judge totals, which are rank-equivalent to the means. The thesis
reports the values produced here.
Exploratory same-SOP judge-generator interaction diagnostics for GPT and Claude.
Holm correction across the two tests.
"""
import pandas as pd
from scipy.stats import wilcoxon
from _analysis_utils import OUT, load_summary, holm_adjust, rank_biserial_paired

df = load_summary()
tests = []

for generator, own_judge, other_judges in [
    ("GPT","GPT",["CLAUDE","GEMINI"]),
    ("CLAUDE","CLAUDE",["GPT","GEMINI"]),
]:
    sub = df[df.generator==generator]
    wide = sub.pivot(index="anonymous_id",columns="judge",values="total_score")
    own = wide[own_judge].to_numpy(float)
    other = wide[other_judges].mean(axis=1).to_numpy(float)
    # test on integers (2*own vs sum of the other two) to avoid float tie noise
    own2 = 2*wide[own_judge].to_numpy(int)
    other2 = wide[other_judges].sum(axis=1).to_numpy(int)
    w = wilcoxon(own2,other2,alternative="two-sided",zero_method="wilcox",method="auto")
    rrb,nz = rank_biserial_paired(own,other)
    diffs = own-other
    tests.append({
        "generator":generator,"own_judge":own_judge,
        "own_mean":own.mean(),"other_two_mean":other.mean(),
        "mean_difference":diffs.mean(),"median_difference":float(pd.Series(diffs).median()),
        "positive":int((diffs>0).sum()),"zero":int((diffs==0).sum()),"negative":int((diffs<0).sum()),
        "T":float(w.statistic),"raw_p":float(w.pvalue),"r_rb":rrb,
        "n_pairs_total":len(own),"n_nonzero_pairs":nz,
    })
adj = holm_adjust([r["raw_p"] for r in tests])
for r,a in zip(tests,adj):
    r["holm_p"] = float(a)
    print(r)
pd.DataFrame(tests).to_csv(OUT / "same_family_judge_diagnostics.csv", index=False)
