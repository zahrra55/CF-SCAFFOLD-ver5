# Code index

Each script is self-contained and was run as a single Kaggle notebook cell. All
of them append per-seed results to a CSV in `/kaggle/working` and skip work
already recorded there, so an interrupted session can be resumed.

---

## Part I: classification

| Script | Produces | Documented in |
|---|---|---|
| `pillar1_classification/full_pipeline.py` | Final canonical pipeline: corrected features (17), patient-grouped classification, radiologist-free delineation by a U-Net localiser, the eight-row federated ablation (the eighth row is real SCAFFOLD), and the leakage check |
| `legacy/full_pipeline_v1_uncorrected_features.py` | Earlier pipeline (16 features, four feature defects). Kept only so that the superseded 0.919 figure can be traced |
| `pillar1_classification/stage14_fb_geometry_confound.py` | Does Fb measure the tumour type or the slice geometry? |
| `pillar1_classification/stage14b_area_matched.py` | Fractal descriptors compared inside bins of head-section area |
| `pillar1_classification/stage15_wallis_buvat_shortcut_reliance.py` | Head-to-head with Wallis and Buvat (2022); shortcut reliance of each feature family |
| `pillar1_classification/stage16_radiologist_free_tumour_features.py` | Radiologist-free classification with the tumour-region features only | [`docs/01`](../docs/01_pillar1_classification.md), [`docs/08`](../docs/08_later_experiments.md) |
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

## Part II: federated

Numbered in the order the experiments were run.

| Script | Produces | Results file | Documented in |
|---|---|---|---|
| `stage1_ablation.py` | Which of six interventions stabilises training | `results/tables/stability_ablation.csv` | [`docs/02`, Stage 1](../docs/02_pillar2_federated.md) |
| `stage1b_statistics.py` | Paired tests over the ablation log | n/a | [`docs/02`, Stage 1](../docs/02_pillar2_federated.md) |
| `stage3_fedrs_comparison.py` | Masking against FedRS at three settings | n/a | [`docs/02`, Stage 3](../docs/02_pillar2_federated.md) |
| `stage4_brisc_external.py` | External validation on BRISC 2025 | `results/tables/brisc_external_validation.csv` | [`docs/02`, Stage 4](../docs/02_pillar2_federated.md) |
| `stage5_alpha_sweep.py` | Benefit as a function of local class absence | `results/tables/alpha_sweep.csv` | [`docs/02`, Stage 5](../docs/02_pillar2_federated.md) |
| `stage6_reviewer_proofing.py` | Logit calibration, centralised bound, residual backbone | `figshare_main_comparison.csv`, `centralized_bound.csv`, `resnet_backbone.csv` | [`docs/02`, Stage 6](../docs/02_pillar2_federated.md) |
| `stage7_bio_contrastive.py` | Contrastive term anchored on fractal descriptors (null) | `results/tables/bio_contrastive_null.csv` | [`docs/03`, §3](../docs/03_withdrawn_and_null.md) |
| `stage9_frozen_backbone.py` | Frozen ImageNet ResNet-18, Figshare | n/a | [`docs/08`, Stage 9](../docs/08_later_experiments.md) |
| `stage9b_frozen_backbone_brisc.py` | Frozen ResNet-18, BRISC | n/a | [`docs/08`, Stage 9B](../docs/08_later_experiments.md) |
| `stage13_real_scaffold.py` | Real SCAFFOLD control variates vs FedAvg, criterion fixed before the run | `results/tables/real_scaffold_stage13.csv` | [`docs/08`, Stage 13](../docs/08_later_experiments.md) |

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


## Final stages: fractal-chaos descriptors and guided federated learning

| Script | Purpose | Result tables | Documentation |
|---|---|---|---|
| `fractal_chaos_guided_fl/stage17_fractal_chaos_descriptors.py` | SampEn2D added; descriptor-based federated learning; BraTS grading | `stage17_*.csv` | [`docs/08`, Stage 17](../docs/08_later_experiments.md) |
| `fractal_chaos_guided_fl/stage17b_same_split_threshold_mask_checks.py` | Same-split damage test (failed rule), in-fold threshold (exploratory), mask normalisation check | `stage17b_*.csv` | [`docs/08`, Stage 17B](../docs/08_later_experiments.md) |
| `fractal_chaos_guided_fl/stage18_matched_heterogeneity_damage.py` | Matched design: heterogeneity damage, CNN vs descriptors, three alpha levels | `stage18_matched_damage.csv` | [`docs/08`, Stage 18](../docs/08_later_experiments.md) |
| `fractal_chaos_guided_fl/stage18b_fractal_attribution.py` | Is the robustness due to the fractal-chaos family? | `stage18b_fractal_attribution.csv` | [`docs/08`, Stage 18B](../docs/08_later_experiments.md) |
| `fractal_chaos_guided_fl/stage19_final_extension.py` | Final extension (v1.1; statistics superseded by Stage 20): grayscale lacunarity, FuzEn2D, boundary FD, glioma vs non-glioma after segmentation, BraTS sub-regions, federated arm | `stage19_*.csv` | [`docs/08`, Stage 19](../docs/08_later_experiments.md) |
| `fractal_chaos_guided_fl/stage20_statistical_corrections.py` | **Final figures.** Corrected repeated-CV t-test, nested selection, strict tumour-only set, glioma vs non-glioma under intention-to-treat, BraTS with all 369 cases, Stage 18 rerun with a bootstrap over test patients | `stage20_*.csv` | [`docs/08`, Stage 20](../docs/08_later_experiments.md) |
