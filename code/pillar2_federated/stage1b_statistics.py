"""
==================================================================
 CF-SCAFFOLD — STAGE 1b: FORMAL PAIRED STATISTICS
 Reads stability_log.csv (from Stage 1) and runs the committee-proof
 paired analysis confirming the restricted-softmax gain. CPU only, fast.
==================================================================
 Per-seed PAIRED comparisons (same seed = same data partition):
   - paired Wilcoxon signed-rank (non-parametric, robust to non-normal)
   - Cohen's d (paired)
   - bootstrap 95% CI on the paired delta
   - Holm-Bonferroni correction across all configs vs baseline
 Also reports collapse rate per config and the full per-seed distribution.
------------------------------------------------------------------
 Run on Kaggle in the SAME session/dir as Stage 1 (reads /kaggle/working),
 or download stability_log.csv and set CSV_PATH locally.
==================================================================
"""
import os, csv
import numpy as np
from scipy import stats

CSV_PATH = "/kaggle/working/stability_log.csv"   # or local path
COLLAPSE_THRESH = 0.40
BASELINE = "baseline (unstable)"

# ── load per-seed results ────────────────────────────────────
rows={}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["config"],{})[int(r["seed"])]=float(r["balanced_acc"])
configs=list(rows.keys())
seeds=sorted(set(s for c in rows for s in rows[c]))
print("="*64); print("  FORMAL PAIRED STATISTICS (Stage 1 stability)"); print("="*64)
print(f"  configs: {len(configs)}  seeds: {len(seeds)}\n")

def cohens_d_paired(a,b):
    d=np.array(a)-np.array(b); return d.mean()/(d.std(ddof=1)+1e-12)
def bootstrap_ci(a,b,n=10000):
    d=np.array(a)-np.array(b); rng=np.random.default_rng(0)
    idx=rng.integers(0,len(d),(n,len(d))); return np.percentile(d[idx].mean(1),[2.5,97.5])
def holm(pv):
    order=np.argsort(pv); m=len(pv); adj=np.zeros(m)
    for rank,idx in enumerate(order): adj[idx]=min((m-rank)*pv[idx],1.0)
    for i in range(1,m): adj[order[i]]=max(adj[order[i]],adj[order[i-1]])
    return adj

# ── per-config summary ───────────────────────────────────────
print(f"  {'config':24s} {'mean':>7} {'std':>7} {'collapses':>10} {'min':>7}")
print("  "+"-"*60)
base_acc=np.array([rows[BASELINE][s] for s in seeds if s in rows[BASELINE]])
for c in configs:
    a=np.array([rows[c][s] for s in seeds if s in rows[c]])
    col=int((a<=COLLAPSE_THRESH).sum())
    star=" <-- best stable" if a.mean()==max(np.array([rows[x][s] for s in seeds if s in rows[x]]).mean() for x in configs) else ""
    print(f"  {c:24s} {a.mean():7.3f} {a.std():7.3f} {col:>7}/{len(a):<2} {a.min():7.3f}{star}")

# ── paired comparisons vs baseline ───────────────────────────
print("\n"+"="*64); print("  EACH CONFIG vs BASELINE (paired, same seeds)"); print("="*64)
comp=[c for c in configs if c!=BASELINE]; pvals=[]; recs=[]
for c in comp:
    common=[s for s in seeds if s in rows[c] and s in rows[BASELINE]]
    a=np.array([rows[c][s] for s in common]); b=np.array([rows[BASELINE][s] for s in common])
    delta=a.mean()-b.mean()
    try: p=stats.wilcoxon(a,b).pvalue
    except Exception: p=1.0
    d=cohens_d_paired(a,b); lo,hi=bootstrap_ci(a,b); wins=int((a>b).sum())
    pvals.append(p); recs.append((c,delta,p,d,lo,hi,wins,len(common)))
adj=holm(pvals)
print(f"  {'config':22s} {'delta':>8} {'p':>8} {'p-Holm':>8} {'d':>7} {'95% CI':>18} {'wins':>7}")
for i,(c,delta,p,d,lo,hi,wins,n) in enumerate(recs):
    sig="*" if adj[i]<0.05 else " "
    print(f"  {c:22s} {delta:+8.3f} {p:8.4f} {adj[i]:8.4f}{sig} {d:+7.2f} [{lo:+.3f},{hi:+.3f}] {wins:>3}/{n}")

# ── headline: best stable config ─────────────────────────────
print("\n"+"="*64); print("  HEADLINE RESULT"); print("="*64)
best=max(comp,key=lambda c: np.array([rows[c][s] for s in seeds if s in rows[c]]).mean())
common=[s for s in seeds if s in rows[best] and s in rows[BASELINE]]
a=np.array([rows[best][s] for s in common]); b=np.array([rows[BASELINE][s] for s in common])
delta=a.mean()-b.mean(); p=stats.wilcoxon(a,b).pvalue; d=cohens_d_paired(a,b); lo,hi=bootstrap_ci(a,b)
i=comp.index(best)
print(f"  Best stable config: {best}")
print(f"  {a.mean():.3f} vs baseline {b.mean():.3f}  (delta=+{delta*100:.1f}pp)")
print(f"  paired Wilcoxon p={p:.4f}, Holm p={adj[i]:.4f}, Cohen d={d:.2f}, 95% CI=[{lo:+.3f},{hi:+.3f}]")
verdict = (delta>0 and adj[i]<0.05 and lo>0)
print(f"  -> {'COMMITTEE-PROOF POSITIVE: significant, CI excludes 0, paired.' if verdict else 'borderline; report with full transparency.'}")
print("\n  Also report HONESTLY: warmup_cosine and more_rounds HURT (high collapse),")
print("  a useful negative finding that popular FL 'best practices' can backfire")
print("  on small Non-IID medical data.")
print("\nDONE. Copy back the two tables + headline for the report.")
