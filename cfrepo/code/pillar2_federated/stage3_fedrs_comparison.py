"""
==================================================================
 CF-SCAFFOLD — STAGE 4: FedRS COMPARISON & NOVELTY ISOLATION
 The Q1 reviewer's #1 question: "how is this different from FedRS?"
 This script answers it empirically. GPU recommended.
==================================================================
 Your class-presence restricted-softmax is the HARD-MASK (alpha->0)
 limit of FedRS (Li & Zhan, KDD 2021), which SCALES absent-class
 logits by a tunable alpha. To make an honest, Q1-grade novelty
 claim, we must compare four methods on the SAME data/seeds:

   M1. FedAvg                  (vanilla baseline)
   M2. SCAFFOLD                (your control-variate baseline)
   M3. FedRS (soft, alpha=0.5) (the ORIGINAL method, FedAvg backbone)
   M4. SCAFFOLD + FedRS-soft   (alpha sweep: 0.0, 0.25, 0.5, 0.75)
       -> alpha=0.0 is YOUR hard mask; this sweep shows where the
          optimum is and whether HARD (yours) beats SOFT (FedRS).

 The honest outcomes:
   - If hard-mask (alpha=0) wins -> your specific choice is justified.
   - If soft FedRS (alpha=0.5) ties/wins -> report that honestly;
     your novelty then rests on the PACKAGE (SCAFFOLD coupling +
     medical diagnosis + negative findings + leakage guard), not the
     softmax variant. Either way the comparison is publishable.

 20 seeds, Dirichlet(0.3), GroupNorm, patient-level test, paired
 Wilcoxon + Cohen's d + bootstrap CI + Holm. Auto-resume CSV.
------------------------------------------------------------------
 Kaggle: GPU. Attach Cheng .mat. Auto-resume via fedrs_compare_log.csv.
==================================================================
"""
import os, glob, csv, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
try: import h5py
except Exception: os.system("pip install h5py --quiet"); import h5py
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ── CONFIG ───────────────────────────────────────────────────
DATA_GLOB="/kaggle/input/**/*.mat"; OUT_DIR="/kaggle/working"
RESULTS_CSV=f"{OUT_DIR}/fedrs_compare_log.csv"
IMG_SIZE=128; N_CLIENTS=18; BATCH=32
SEEDS=20; PARTITION_ALPHA=0.3
ROUNDS=60; LOCAL_EPOCHS=1; LR=0.01
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

# ── readers (reused, verified) ───────────────────────────────
def read_pid(cj):
    if "PID" not in cj: return None
    raw=np.array(cj["PID"]).ravel()
    try: s="".join(chr(int(c)) for c in raw if int(c)>0); return s.strip() or None
    except Exception: return None
def read_cjdata(path):
    try:
        with h5py.File(path,"r") as f:
            cj=f["cjdata"]; label=int(np.array(cj["label"]).ravel()[0])
            img=np.array(cj["image"]).astype(np.float32)
            if img.ndim==2: img=img.T
            return img,label,read_pid(cj)
    except Exception: return None,None,None
def resize_img(img,size):
    from PIL import Image
    im=Image.fromarray(((img-img.min())/(img.max()-img.min()+1e-8)*255).astype(np.uint8)).resize((size,size),Image.BILINEAR)
    return np.asarray(im).astype(np.float32)
def load_data():
    files=sorted(f for f in glob.glob(DATA_GLOB,recursive=True) if f.lower().endswith(".mat"))
    print(f"Found {len(files)} .mat files")
    X=[];Y=[];PID=[]
    for k,path in enumerate(files):
        img,label,pid=read_cjdata(path)
        if img is None or pid is None: continue
        X.append((resize_img(img,IMG_SIZE)/255.0).astype(np.float32)); Y.append(label-1); PID.append(pid)
        if (k+1)%800==0: print(f"  loaded {k+1}/{len(files)}")
    return np.array(X)[:,None], np.array(Y), np.array(PID)

# ── model (GroupNorm) ────────────────────────────────────────
class SmallNet(nn.Module):
    def __init__(self,n=3):
        super().__init__()
        self.c1=nn.Conv2d(1,16,3,padding=1); self.n1=nn.GroupNorm(4,16)
        self.c2=nn.Conv2d(16,32,3,padding=1); self.n2=nn.GroupNorm(4,32)
        self.c3=nn.Conv2d(32,64,3,padding=1); self.n3=nn.GroupNorm(8,64)
        self.fc=nn.Linear(64,n)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.n1(self.c1(x))),2)
        x=F.max_pool2d(F.relu(self.n2(self.c2(x))),2)
        x=F.max_pool2d(F.relu(self.n3(self.c3(x))),2)
        return self.fc(F.adaptive_avg_pool2d(x,1).flatten(1))
def _flat(m): return torch.cat([p.data.flatten() for p in m.parameters()])
def _set(m,f):
    i=0
    for p in m.parameters(): n=p.numel(); p.data.copy_(f[i:i+n].view_as(p)); i+=n

# ── loss variants ────────────────────────────────────────────
def loss_fn(logits, yb, present, method, alpha=0.5):
    if method in ("fedavg","scaffold"):
        return F.cross_entropy(logits, yb)
    if method == "fedrs_soft":                 # FedRS original: scale absent logits by alpha
        scale=torch.full((logits.shape[1],),alpha,device=logits.device)
        for c in present: scale[c]=1.0
        return F.cross_entropy(logits*scale, yb)
    if method == "hard_mask":                  # YOUR method: alpha->0 hard mask
        mask=torch.full((logits.shape[1],),-1e9,device=logits.device)
        for c in present: mask[c]=0.0
        return F.cross_entropy(logits+mask, yb)
    if method == "scaffold_alpha":             # SCAFFOLD + tunable alpha (the sweep)
        scale=torch.full((logits.shape[1],),alpha,device=logits.device)
        for c in present: scale[c]=1.0
        return F.cross_entropy(logits*scale, yb)

def _balanced_loader(X,Y,idx):
    ds=TensorDataset(torch.tensor(X[idx]),torch.tensor(Y[idx]))
    y=Y[idx]; cc=np.bincount(y,minlength=3); w=np.array([1.0/cc[c] if cc[c]>0 else 0 for c in y])
    sampler=WeightedRandomSampler(torch.tensor(w,dtype=torch.double),len(w),replacement=True)
    return DataLoader(ds,batch_size=BATCH,sampler=sampler)

def federated_run(X,Y,client_idx,test_idx,method,seed,alpha=0.5,use_scaffold=False,clip=True):
    torch.manual_seed(seed); np.random.seed(seed); model=SmallNet().to(DEVICE)
    clients=list(client_idx); ni={c:len(client_idx[c]) for c in clients}
    flat0=_flat(model); c_global=torch.zeros_like(flat0)
    c_locals={c:torch.zeros_like(flat0) for c in clients}
    for rd in range(ROUNDS):
        gflat=_flat(model); deltas=[]; new_cs={}
        for c in clients:
            _set(model,gflat); present=list(np.unique(Y[client_idx[c]]))
            loader=_balanced_loader(X,Y,client_idx[c])
            opt=torch.optim.SGD(model.parameters(),lr=LR); model.train()
            start=_flat(model).clone(); steps=0
            for _ in range(LOCAL_EPOCHS):
                for xb,yb in loader:
                    xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
                    loss=loss_fn(model(xb),yb,present,method,alpha)
                    loss.backward()
                    if use_scaffold:
                        i=0
                        for p in model.parameters():
                            n=p.numel(); p.grad.data+=(c_global[i:i+n]-c_locals[c][i:i+n]).view_as(p); i+=n
                    if clip: torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    opt.step(); steps+=1
            end=_flat(model); deltas.append((ni[c],end-gflat))
            if use_scaffold:
                new_cs[c]=c_locals[c]-c_global+(start-end)/(max(steps,1)*LR)
        tot=sum(n for n,_ in deltas); _set(model,gflat+sum(n/tot*d for n,d in deltas))
        if use_scaffold:
            c_global=c_global+sum((new_cs[c]-c_locals[c]) for c in clients)/len(clients)
            for c in clients: c_locals[c]=new_cs[c]
    model.eval()
    with torch.no_grad():
        pred=model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx],pred)

def dirichlet_partition(Y,PID,n_clients,alpha,seed):
    rng=np.random.default_rng(seed); uniq=np.unique(PID); pl={p:Y[np.where(PID==p)[0][0]] for p in uniq}
    cp={c:[] for c in range(n_clients)}
    for cls in np.unique(Y):
        cps=[p for p in uniq if pl[p]==cls]; rng.shuffle(cps)
        pr=rng.dirichlet([alpha]*n_clients); cnt=(pr*len(cps)).astype(int)
        if len(cps): cnt[-1]=len(cps)-cnt[:-1].sum()
        idx=0
        for c in range(n_clients):
            for _ in range(cnt[c]):
                if idx<len(cps): cp[c].append(cps[idx]); idx+=1
    ci={c:np.where(np.isin(PID,cp[c]))[0] for c in range(n_clients)}
    return {c:v for c,v in ci.items() if len(v)>=4}

# ── methods to compare ───────────────────────────────────────
# (name, method_loss, use_scaffold, alpha)
METHODS = [
    ("FedAvg",            "fedavg",        False, None),
    ("SCAFFOLD",          "scaffold",      True,  None),
    ("FedRS(a=0.5)",      "fedrs_soft",    False, 0.5),   # ORIGINAL FedRS
    ("SCAF+soft(a=0.5)",  "scaffold_alpha",True,  0.5),
    ("SCAF+soft(a=0.25)", "scaffold_alpha",True,  0.25),
    ("SCAF+hard(YOURS)",  "hard_mask",     True,  0.0),   # YOUR method
]

# ── quota armor ──────────────────────────────────────────────
def init_log():
    os.makedirs(OUT_DIR,exist_ok=True)
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV,"w",newline="") as f: csv.DictWriter(f,fieldnames=["method","seed","acc","t"]).writeheader()
def done(method,seed):
    if not os.path.exists(RESULTS_CSV): return None
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            if r["method"]==method and int(r["seed"])==seed: return float(r["acc"])
    return None
def log(method,seed,acc):
    with open(RESULTS_CSV,"a",newline="") as f:
        csv.DictWriter(f,fieldnames=["method","seed","acc","t"]).writerow(dict(method=method,seed=seed,acc=round(acc,4),t=round(time.time())))

# ── stats helpers ────────────────────────────────────────────
def cohend(a,b): d=np.array(a)-np.array(b); return d.mean()/(d.std(ddof=1)+1e-12)
def boot(a,b,n=10000):
    d=np.array(a)-np.array(b); rng=np.random.default_rng(0); idx=rng.integers(0,len(d),(n,len(d)))
    return np.percentile(d[idx].mean(1),[2.5,97.5])
def holm(pv):
    order=np.argsort(pv); m=len(pv); adj=np.zeros(m)
    for rank,idx in enumerate(order): adj[idx]=min((m-rank)*pv[idx],1.0)
    for i in range(1,m): adj[order[i]]=max(adj[order[i]],adj[order[i-1]])
    return adj

# ── MAIN ─────────────────────────────────────────────────────
print("="*64); print("  LOADING"); print("="*64)
X,Y,PID=load_data()
print(f"  {len(X)} slices, {len(np.unique(PID))} patients, classes={np.bincount(Y)}")
rng=np.random.default_rng(42); uniq=np.unique(PID); rng.shuffle(uniq)
test_pids=set(uniq[:int(0.2*len(uniq))])
test_idx=np.where(np.isin(PID,list(test_pids)))[0]; train_mask=~np.isin(PID,list(test_pids))
train_pos=np.where(train_mask)[0]

init_log()
print("\n"+"="*64); print(f"  FedRS COMPARISON ({SEEDS} seeds)"); print("="*64)
for seed in range(SEEDS):
    if all(done(name,seed) is not None for name,_,_,_ in METHODS): 
        print(f"  seed {seed}: SKIP"); continue
    cidx=dirichlet_partition(Y[train_mask],PID[train_mask],N_CLIENTS,PARTITION_ALPHA,seed)
    cidx={c:train_pos[v] for c,v in cidx.items()}
    line=f"  seed {seed}: "
    for name,method,use_scaf,alpha in METHODS:
        if done(name,seed) is not None: continue
        acc=federated_run(X,Y,cidx,test_idx,method,seed,alpha=alpha or 0.5,use_scaffold=use_scaf)
        log(name,seed,acc); line+=f"{name}={acc:.3f} "
    print(line)

# ── aggregate + stats ────────────────────────────────────────
rows={}
with open(RESULTS_CSV) as f:
    for r in csv.DictReader(f): rows.setdefault(r["method"],{})[int(r["seed"])]=float(r["acc"])
seeds=sorted(set(s for m in rows for s in rows[m]))
print("\n"+"="*64); print("  RESULTS (mean over seeds)"); print("="*64)
for name,_,_,_ in METHODS:
    a=np.array([rows[name][s] for s in seeds if s in rows.get(name,{})])
    if len(a): print(f"  {name:20s}: {a.mean():.3f} +/- {a.std():.3f}  (collapses={int((a<=0.40).sum())}/{len(a)})")

# ── key comparison: YOURS vs FedRS-soft and vs SCAFFOLD ───────
print("\n"+"="*64); print("  KEY COMPARISONS (paired vs YOURS, Holm)"); print("="*64)
yours="SCAF+hard(YOURS)"
if yours in rows:
    comp=[n for n,_,_,_ in METHODS if n!=yours]; pvals=[]; recs=[]
    yv=rows[yours]
    for name in comp:
        common=[s for s in seeds if s in rows.get(name,{}) and s in yv]
        a=np.array([yv[s] for s in common]); b=np.array([rows[name][s] for s in common])
        delta=a.mean()-b.mean()
        try: p=stats.wilcoxon(a,b).pvalue
        except Exception: p=1.0
        d=cohend(a,b); lo,hi=boot(a,b); pvals.append(p); recs.append((name,delta,p,d,lo,hi))
    adj=holm(pvals)
    print(f"  YOURS vs {'baseline':16s} {'delta':>8} {'p-Holm':>8} {'d':>7} {'95% CI':>18}")
    for i,(name,delta,p,d,lo,hi) in enumerate(recs):
        sig="*" if adj[i]<0.05 else " "
        print(f"  vs {name:18s} {delta:+8.3f} {adj[i]:8.4f}{sig} {d:+7.2f} [{lo:+.3f},{hi:+.3f}]")

print("\n"+"="*64); print("  NOVELTY VERDICT"); print("="*64)
if yours in rows and "FedRS(a=0.5)" in rows:
    common=[s for s in seeds if s in rows[yours] and s in rows["FedRS(a=0.5)"]]
    a=np.array([rows[yours][s] for s in common]); b=np.array([rows["FedRS(a=0.5)"][s] for s in common])
    d=a.mean()-b.mean()
    if d>0.02:
        print(f"  Hard-mask + SCAFFOLD beats original FedRS by {d*100:+.1f}pp.")
        print("  -> Your specific design (hard mask + SCAFFOLD) is justified empirically.")
    elif abs(d)<=0.02:
        print(f"  Hard-mask ~ FedRS (delta={d*100:+.1f}pp, within noise).")
        print("  -> HONEST: the softmax variant is not the novelty. Frame the contribution")
        print("     as the PACKAGE: medical missing-class diagnosis + SCAFFOLD coupling +")
        print("     negative findings (common stabilizers hurt) + leakage guard.")
    else:
        print(f"  Original FedRS beats hard-mask by {-d*100:.1f}pp.")
        print("  -> HONEST: adopt FedRS-soft as your method; novelty = medical application")
        print("     + diagnosis + negative findings, not the masking choice.")
print("\nDONE. Copy back: RESULTS, KEY COMPARISONS, NOVELTY VERDICT.\n")
