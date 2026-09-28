# Code index

Each script is self-contained and was run as a single Kaggle notebook cell. All
of them append per-seed results to a CSV in `/kaggle/working` and skip work
already recorded there, so an interrupted session can be resumed.

---

## Part I — classification

| Script | Produces | Documented in |
|---|---|---|
| `pillar1_classification/full_pipeline.py` | Canonical pipeline: feature extraction, patient-grouped classification, the eight-row federated ablation (the eighth row is real SCAFFOLD), and the leakage check | [`docs/01`](../docs/01_pillar1_classification.md), [`docs/08`](../docs/08_later_experiments.md) |
| `pillar1_classification/stage10_descriptor_tests.py` | Single-split descriptor tests (superseded by 10B, kept for the record) | [`docs/08`, Stage 10](../docs/08_later_experiments.md) |
| `pillar1_classification/stage10b_repeated_splits_and_leakage.py` | C vs Ft, Fb over ten splits; leakage inflation measured | [`docs/08`, Stage 10B](../docs/08_later_experiments.md) |
| `pillar1_classification/stage11_active_contour.py` | Automatic segmentation by seeded morphological Chan–Vese | [`docs/08`, Stage 11](../docs/08_later_experiments.md) |
| `pillar1_classification/stage12_dbc_audit_and_symmetry_seed.py` | Audit of an alternative DBC script; bilateral-symmetry seed | [`docs/08`, Stage 12](../docs/08_later_experiments.md) |

The pipeline has three switches at the top: `RUN_PILLAR_1` for classification,
`RUN_PILLAR_2` for the federated ablation, and `RUN_LEAKAGE_GUARD` for the
protocol check. It expects the Figshare dataset. In the ablation, `scaffold=False`
(the default for the first seven rows) reproduces the original code path exactly;
only the row `+restricted_softmax+SCAFFOLD` enables control variates.

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
| `stage9_frozen_backbone.py` | Frozen ImageNet ResNet-18, Figshare | — | [`docs/08`, Stage 9](../docs/08_later_experiments.md) |
| `stage9b_frozen_backbone_brisc.py` | Frozen ResNet-18, BRISC | — | [`docs/08`, Stage 9B](../docs/08_later_experiments.md) |
| `stage13_real_scaffold.py` | Real SCAFFOLD control variates vs FedAvg, preregistered criterion | `results/tables/real_scaffold_stage13.csv` | [`docs/08`, Stage 13](../docs/08_later_experiments.md) |

Stages 1–7 were numbered when this repository was first assembled; Stages 9–13
keep the numbers used in the session record so that scripts, logs and
documentation match. There is no gap in the work, only in the numbering.

`stage6_reviewer_proofing.py` has three switches, `RUN_PART_A`, `RUN_PART_B` and
`RUN_PART_C`. Parts A and B fit one session together; Part C is best run in a
second session, and the data cache makes it start immediately.

---

## Shared

| Script | Purpose |
|---|---|
| `shared/leakage_guard.py` | Scores an image-blind predictor under a support/query split. Run before any per-patient adaptation experiment. |
| `shared/brisc_structure_probe.py` | Prints the actual directory layout of an attached dataset. Written after a mount layout differed from the documented one. |
| `audits/colleague_pipeline_resolution_fix.py` | Corrected version of an external script (features at 128 × 128, brain mask with opening). Audit record, not part of the canonical pipeline. |

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
the federated code, except inside the frozen, pretrained ResNet-18 of Stage 9,
where its statistics are fixed and never averaged.

**Aggregation.** Weighted FedAvg unless a script says otherwise. SCAFFOLD control
variates are used in `stage3_fedrs_comparison.py`, `stage13_real_scaffold.py`
and the eighth ablation row of `full_pipeline.py`.

**Statistics.** Paired Wilcoxon, Cohen's `d` for paired samples, a
10,000-resample bootstrap interval, and Holm correction across the baselines in
each comparison family.
