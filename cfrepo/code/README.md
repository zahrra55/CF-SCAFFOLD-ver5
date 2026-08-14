# Code index

Each script is self-contained and was run as a single Kaggle notebook cell. All
of them append per-seed results to a CSV in `/kaggle/working` and skip work
already recorded there, so an interrupted session can be resumed.

---

## Part I — classification

| Script | Produces | Documented in |
|---|---|---|
| `pillar1_classification/full_pipeline.py` | Feature extraction, patient-grouped classification, and the leakage check, in one file | [`docs/01`](../docs/01_pillar1_classification.md) |

The pipeline has three switches at the top: `RUN_PILLAR_1` for classification,
`RUN_PILLAR_2` for the federated ablation, and `RUN_LEAKAGE_GUARD` for the
protocol check. It expects the Figshare dataset.

---

## Part II — federated

Numbered in the order the experiments were run.

| Script | Produces | Results file | Documented in |
|---|---|---|---|
| `stage1_ablation.py` | Which of six interventions stabilises training | `results/tables/stability_ablation.csv` | [`docs/02`, Stage 1](../docs/02_pillar2_federated.md) |
| `stage1b_statistics.py` | Paired tests over the ablation log | — | [`docs/02`, Stage 1](../docs/02_pillar2_federated.md) |
| `stage3_fedrs_comparison.py` | Masking against FedRS at three settings | — | [`docs/02`, Stage 3](../docs/02_pillar2_federated.md) |
| `stage4_brisc_external.py` | External validation on BRISC 2025 | `results/tables/brisc_external_validation.csv` | [`docs/02`, Stage 4](../docs/02_pillar2_federated.md) |
| `stage5_alpha_sweep.py` | Benefit as a function of local class absence | `results/tables/alpha_sweep.csv` | [`docs/02`, Stage 5](../docs/02_pillar2_federated.md) |
| `stage6_reviewer_proofing.py` | Logit calibration, centralised bound, residual backbone | `figshare_main_comparison.csv`, `centralized_bound.csv`, `resnet_backbone.csv` | [`docs/02`, Stage 6](../docs/02_pillar2_federated.md) |
| `stage7_bio_contrastive.py` | Contrastive term anchored on fractal descriptors (null) | `results/tables/bio_contrastive_null.csv` | [`docs/03`, §3](../docs/03_withdrawn_and_null.md) |

`stage6_reviewer_proofing.py` has three switches, `RUN_PART_A`, `RUN_PART_B` and
`RUN_PART_C`. Parts A and B fit one session together; Part C is best run in a
second session, and the data cache makes it start immediately.

---

## Shared

| Script | Purpose |
|---|---|
| `shared/leakage_guard.py` | Scores an image-blind predictor under a support/query split. Run before any per-patient adaptation experiment. |
| `shared/brisc_structure_probe.py` | Prints the actual directory layout of an attached dataset. Written after a mount layout differed from the documented one. |

---

## Conventions used throughout

**Reading the data.** The Figshare files are MATLAB v7.3, so `h5py` is used
rather than `scipy.io.loadmat`, arrays are transposed on read, and the patient
identifier is decoded from character codes.

**Finding the data.** `os.walk` over `/kaggle/input` rather than a `glob`
pattern, because one mount layout defeated `glob` with `**`.

**Seeding.** One seed fixes the partition, the split, the initialisation and the
batch order together, so any two methods at the same seed face identical
conditions and comparisons are paired.

**Normalisation.** GroupNorm in every model. No BatchNorm appears anywhere in
the federated code.

**Statistics.** Paired Wilcoxon, Cohen's `d` for paired samples, a
10,000-resample bootstrap interval, and Holm correction across the baselines in
each comparison family.
