"""
Final-phase human validation (Section 4.4.8 / 5.8).
Inputs (this folder): human_ratings_anonymised.csv (2 raters x 12 SOPs x 8 criteria,
exported from the Google Form; names removed), hv_key.csv (HV_xx -> anonymous SOP id),
and ../FINAL_EVALUATION_RESULTS/CONSOLIDATED/evaluation_summary_144.csv.
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wilcoxon_integer_rerun"))
from _analysis_utils import load_summary, holm_adjust, CRITERIA, JUDGES

HERE = Path(__file__).resolve().parent
C = [f"C{i}" for i in range(1, 9)]
hum = pd.read_csv(HERE / "human_ratings_anonymised.csv")
key = pd.read_csv(HERE / "hv_key.csv")
ev = load_summary()

def kalpha_ordinal(M):  # units x raters, integer 1..5, no missing values
    o = np.zeros((5, 5))
    for row in M:
        m = len(row)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[row[i]-1, row[j]-1] += 1/(m-1)
    n = o.sum(1); N = n.sum(); d = np.zeros((5, 5))
    for a in range(5):
        for b in range(5):
            lo, hi = min(a, b), max(a, b)
            d[a, b] = (n[lo:hi+1].sum() - (n[a]+n[b])/2)**2
    De = (np.outer(n, n)*d).sum()/(N-1)
    return 1 - (o*d).sum()/De

def icc21(T):  # n x k
    n, k = T.shape; g = T.mean()
    msr = k*((T.mean(1)-g)**2).sum()/(n-1)
    msc = n*((T.mean(0)-g)**2).sum()/(k-1)
    mse = ((T - T.mean(1)[:, None] - T.mean(0)[None, :] + g)**2).sum()/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)

r1 = hum[hum.rater == "R1"].set_index("hv_id").sort_index()
r2 = hum[hum.rater == "R2"].set_index("hv_id").sort_index()

print("RATER DESCRIPTIVES (totals)")
for name, r in [("R1", r1), ("R2", r2)]:
    t = r.total
    print(name, f"mean={t.mean():.2f} sd={t.std(ddof=1):.2f} min={t.min()} max={t.max()}",
          f"share_5={(r[C].to_numpy()==5).mean()*100:.1f}%")

a, b = r1[C].to_numpy().ravel(), r2[C].to_numpy().ravel()
print("\nINTER-RATER (96 criterion ratings)")
print(f"exact={np.mean(a==b)*100:.1f}%  within1={np.mean(abs(a-b)<=1)*100:.1f}%",
      f"QWK={cohen_kappa_score(a, b, weights='quadratic'):.2f}",
      f"alpha_ordinal={kalpha_ordinal(np.c_[a, b]):.2f}",
      f"ICC(2,1)_totals={icc21(np.c_[r1.total, r2.total]):.2f}",
      f"spearman_totals={spearmanr(r1.total, r2.total)[0]:.2f}")

# human mean per SOP vs judges
hm = hum.groupby("hv_id")[C+["total"]].mean().join(key.set_index("hv_id")[["anonymous_id", "generator"]])
crit_map = dict(zip(C, CRITERIA))
rows = []
for j in JUDGES + ["CONSENSUS"]:
    sub = ev if j == "CONSENSUS" else ev[ev.judge == j]
    js = sub.groupby("anonymous_id")[["total_score"]+CRITERIA].mean()
    jt = js.loc[hm.anonymous_id, "total_score"].to_numpy()
    # integer comparison: 2*human sum vs (judge total * number of judges * 2)/... -> use exact
    # rational values scaled to integers to avoid float tie noise
    k = 3 if j == "CONSENSUS" else 1
    h_int = (hum.groupby("hv_id").total.sum().loc[hm.index].to_numpy()*k).astype(int)  # 2*mean*k
    j_int = (sub.groupby("anonymous_id").total_score.sum().loc[hm.anonymous_id].to_numpy()*2).astype(int)
    w = wilcoxon(h_int, j_int, alternative="two-sided", zero_method="wilcox", method="auto")
    rows.append(dict(judge=j, judge_mean=jt.mean(), human_mean=hm.total.mean(),
                     diff=hm.total.mean()-jt.mean(), T=float(w.statistic), raw_p=float(w.pvalue),
                     spearman=spearmanr(hm.total, jt)[0]))
adj = holm_adjust([r["raw_p"] for r in rows])
print("\nHUMAN vs JUDGES (12 SOP totals)")
for r, p in zip(rows, adj):
    print(f"{r['judge']:9s} judge={r['judge_mean']:.2f} human={r['human_mean']:.2f} diff={r['diff']:+.2f}",
          f"T={r['T']:.1f} raw_p={r['raw_p']:.4f} holm_p={p:.3f} rho={r['spearman']:+.2f}")

cons = ev.groupby("anonymous_id")[CRITERIA].mean().loc[hm.anonymous_id]
print("\nCRITERION MEANS human vs consensus")
for c in C:
    print(c, f"{hm[c].mean():.2f} vs {cons[crit_map[c]].mean():.2f}")
print("\nHUMAN MEAN BY GENERATOR")
print(hm.groupby("generator").total.mean().round(2).to_string())
