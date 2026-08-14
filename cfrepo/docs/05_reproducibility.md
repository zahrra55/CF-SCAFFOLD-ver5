# Reproducibility

---

## Environment

All experiments were run on Kaggle notebooks with a GPU accelerator (P100 or
T4). No local GPU was used at any point.

| Component | Version |
|---|---|
| Python | 3.12 (Kaggle default image) |
| PyTorch | Kaggle default at time of run |
| NumPy, SciPy, scikit-learn | Kaggle default |
| h5py | installed by the scripts if absent |
| scikit-image | installed by the scripts if absent |

Package versions are not pinned because the Kaggle image changes over time. The
results reported were produced on the image current at the time each script was
run; the dates are recoverable from the CSV logs, which include timestamps.

---

## What the seed controls

Setting one seed fixes all of the following:

1. The Dirichlet draw that assigns patients to clients.
2. The train/test split of patients.
3. Model initialisation.
4. Batch ordering within each client.

Because a single seed determines the entire configuration, two methods evaluated
at the same seed face identical conditions. This is what makes the comparisons
paired and permits the use of paired statistical tests.

Seeds 0 to 19 were used throughout unless a script states otherwise.

---

## Resumption

Every long-running script appends each completed result to a CSV in
`/kaggle/working` and checks that file before starting any run. A session that
is interrupted, including by the 12-hour Kaggle limit, can be restarted and will
skip everything already recorded.

This is not only a convenience. It also means the reported tables are read back
from disk rather than transcribed by hand, so a number in a table and the number
produced by the run are the same object.

---

## Compute cost

Measured wall-clock times on Kaggle GPU.

| Experiment | Cost |
|---|---|
| Feature extraction (2740 slices) | ~6 min |
| Part I classification, 5-fold with grid search | ~10 min |
| Stage 1 ablation, 7 configurations × 20 seeds | ~4 h |
| Stage 3 FedRS comparison, 4 methods × 20 seeds | ~45 min |
| Stage 4 BRISC external, 4 methods × 20 seeds | ~45 min |
| Stage 5 alpha sweep, 3 × 3 × 8 | ~40 min |
| Stage 6a FedLC comparison, 4 methods × 20 seeds | ~45 min |
| Stage 6b centralised bound, 5 seeds | ~3 min |
| Stage 6c residual network, 4 methods × 10 seeds | ~32 min |
| Stage 7 contrastive, 4 arms × 16 seeds | ~12 h (hit session limit) |

The contrastive experiment is the expensive one because it recomputes fractal
descriptors during data loading and adds a pairwise term to every batch.

---

## Running an experiment

1. Create a Kaggle notebook with the GPU accelerator enabled.
2. Attach the dataset through **Add Input**. Select the entry marked *Dataset*.
   Attaching a notebook of the same name is a mistake that produces
   `Found 0 .mat files`; it happened twice during this project.
3. Paste the script and run.
4. Copy the printed tables back. Nothing else needs to be saved, since the CSV
   in `/kaggle/working` holds the per-seed values.

The loaders search `/kaggle/input` recursively with `os.walk`, so they find the
data regardless of how deeply Kaggle mounts it. An earlier version used
`glob` with `**`, which failed on one mount layout.

---

## Known sources of small variation

Results reproduce to within about 0.005 balanced accuracy between runs of the
same script on the same seeds. Sources:

- cuDNN kernel selection is not deterministic for all operations;
- the Kaggle image may update between sessions;
- `WeightedRandomSampler` draws with replacement, and while the seed fixes the
  sequence, a change in worker behaviour can alter it.

This is why the paper reports means over 20 seeds with dispersion, and why the
one borderline result in the project — logit calibration ahead by 0.76 points at
`p = 0.008` — was not treated as established when a repeat run returned 0.4
points at `p = 0.12`.

---

## Verifying a number in the thesis

For any figure or table:

1. Find the stage in `docs/01_pillar1_classification.md` or
   `docs/02_pillar2_federated.md`.
2. The stage names the script in `code/`.
3. The per-seed values are in `results/tables/`.
4. Running the script on the stated dataset reproduces those values within the
   tolerance above.
