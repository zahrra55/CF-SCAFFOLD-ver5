# CF-SCAFFOLD: Positive Results

October 2026. Companion file: [`10_negative_results.md`](10_negative_results.md).

## How to read this file

The experimental work is frozen. Every figure below comes from runs completed by October 2026 under patient-level evaluation, and can be traced to a script and a result table (Section 6). Where Stage 20, the statistical correction made after an external code review, re-tested a result, its figure replaces the earlier one. Results that were not confirmed, and the comparisons that bound each claim, are recorded in the companion file of negative results.

Each result carries one of four evidence labels:

| Label | Meaning |
| --- | --- |
| Rule met | Decision rule written into the script before the run, and met. The rules were not registered externally, and the same data had been analysed in earlier stages. |
| Repeated | Reproduced across independent runs or repeated splits, without a decision rule |
| Replication | Confirms a published finding with a different method or protocol |
| Exploratory | Analysis decided after seeing earlier results; reported for information only |

Comparisons between descriptor sets use the corrected repeated cross-validation t-test (Nadeau and Bengio, 2003), because repeated splits share patients and ordinary tests overstate significance. Federated comparisons use a bootstrap over test patients. Tests over training seeds are reported only as conditional on the fixed test patients.

All accuracies are balanced accuracies, at patient level unless stated as slice level. The thesis title is *Fractal-Chaos Guided Federated Learning for Brain Tumor MRI Classification*.

## Headline results

Federated learning guided by fractal-chaos descriptors loses 13.9 points less to heterogeneity than a CNN on images, and the fractal-chaos family adds 13.2 points to the descriptor model.

| Result | Value | Evidence |
| --- | --- | --- |
| Heterogeneity damage gap, CNN minus 19-descriptor model (alpha 0.3) | +13.9 points, 95% CI [10.8, 17.1] over test patients | Rule met (Stage 20) |
| Damage from heterogeneity at alpha 0.3 | CNN on images 15.2 points; 19 descriptors 1.3 points | Rule met (Stage 20) |
| Federated learning on the 19 descriptors (alpha 0.3) | 0.856 slice level, 0.923 patient level | Repeated (Stages 19, 20) |
| Contribution of the fractal-chaos family to the descriptor model | +13.2 points, 95% CI [7.2, 19.1], corrected p < 0.0001 | Repeated (Stage 20) |
| Three-class classification, 19 descriptors (set chosen by the rule) | 0.923 ± 0.006 | Repeated (Stage 20) |
| Three-class classification, nested-selection estimate | 0.939 ± 0.038 per outer fold | Repeated (Stage 20) |
| Three-class classification, strictly tumour-only descriptors | 0.736 ± 0.013 | Repeated (Stage 20) |
| Glioma vs non-glioma after automatic segmentation (intention-to-treat) | 0.946, 95% CI [0.915, 0.973] | Repeated (Stage 20) |
| Class-presence correction, federated CNN | 0.649 vs 0.584 for FedAvg under identical settings; equivalent to logit calibration | Repeated (seed-level, conditional on the test patients) |
| Classification without a radiologist, three classes | No detectable cost: -0.6 points, 95% CI [-4.6, +3.7] | Repeated (Stage 16, bootstrap over patients) |

## 1. Fractal-chaos descriptors and three-class classification

The fractal-chaos family is the largest single contributor to classifying glioma, meningioma and pituitary tumours, and its contribution survives the corrected statistical test.

| Finding | Value | Evidence | Stage |
| --- | --- | --- | --- |
| Contribution of the fractal-chaos family (19 descriptors with vs without it) | +13.2 points, 95% CI [7.2, 19.1], corrected p < 0.0001 | Repeated | 20 |
| 19 descriptors, the set chosen by the decision rule | 0.923 ± 0.006 | Repeated | 20 |
| Nested selection among eight candidate sets | 0.939 ± 0.038 per outer fold, against 0.918 for the rule-chosen set on the same folds | Repeated | 20 |
| Fractal-chaos family alone | 0.756 ± 0.014 | Repeated | 20 |
| Pituitary discrimination inside bins of equal head-section area | AUC of C 0.886 and of Ft 0.876, while section area itself fell to 0.518 | Repeated | 14B |

Nested selection chooses the descriptor set inside each outer training fold, so its estimate carries no selection optimism. In all 15 outer folds the chosen set contained grayscale lacunarity. Tested on its own, grayscale lacunarity raised accuracy by 2.38 points, an effect the corrected test did not confirm (p = 0.098); it is therefore listed among the unconfirmed results.

The estimate that is robust to the dataset shortcut is the strictly tumour-only configuration, 0.736, well above chance (0.333). The earlier tumour-only figure (0.829 in Stage 20) included three descriptors that use pixels outside the tumour or mostly encode its size; binary-mask lacunarity of the tumour, for example, correlates with tumour size at -0.98.

## 2. Glioma vs non-glioma, and classification without a radiologist

After automatic segmentation, the descriptors separate gliomas from meningiomas and pituitary tumours at 0.946, about 15 points above what head geometry alone achieves. The task is a proxy for malignant versus predominantly benign tumours.

| Arm (same split, 218 patients) | Patient-level balanced accuracy, 95% CI |
| --- | --- |
| 19 descriptors, radiologist masks | 0.979 [0.958, 0.996] |
| 19 descriptors, U-Net masks, intention-to-treat | 0.946 [0.915, 0.973] |
| Strictly tumour-only descriptors, radiologist masks | 0.871 [0.823, 0.913] |
| Fractal-chaos family, radiologist masks | 0.802 [0.750, 0.852] |
| Head geometry alone (shortcut control) | 0.797 [0.746, 0.847] |

The margin of the automatic pipeline over the shortcut control is +0.149, 95% CI [0.091, 0.209], clearly above zero. The decision rule asked for a lower bound of at least 0.10 and was narrowly missed, which is recorded in the negative results. Under intention-to-treat, a patient whose slices all failed segmentation counts as an error; no patient did. The U-Net Dice over all 2740 slices, failures included, had a mean of 0.710 and a median of 0.821, with 67.5% of slices above 0.7; automatic descriptors could be computed on 2708 slices.

For the three-class task, replacing the radiologist's mask by the U-Net mask caused no detectable loss with the full descriptor set (0.897 to 0.892; 95% CI of the difference -4.6 to +3.7 points; Stage 16, bootstrap over patients).

## 3. Federated learning on images: the class-presence correction

When hospitals lack some tumour types, restricting each hospital's softmax to the classes it holds raises the federated CNN from 0.584 to 0.649 and matches the strongest published correction without a tuned parameter.

| Comparison (Figshare, 18 hospitals, Dirichlet 0.3, 20 seeds) | Balanced accuracy |
| --- | --- |
| FedAvg | 0.584 ± 0.074 |
| FedRS (alpha 0.5) | 0.592 ± 0.072 |
| Logit calibration (FedLC) | 0.653 ± 0.026 |
| Class-presence correction | 0.649 ± 0.028 |
| Centralised on pooled data | 0.705 ± 0.011 |

The correction exceeded FedRS by 5.8 points (Holm p = 0.002) and was statistically equivalent to logit calibration (Holm p = 0.123); these tests compare 20 training seeds on the same test patients, so they are conditional on those patients. On the external BRISC dataset it exceeded FedRS by 5.2 points. Its benefit grew with the share of absent classes: +17.5, +6.3 and +0.2 points at three decreasing levels of skew, which tells a practitioner when the correction is worth applying. Within the stabilised training recipe, no run collapsed to chance in 20 seeds, and the result reproduced in three independent runs.

The mechanism is explained by the gradient of the softmax cross-entropy: for a class a hospital never observes, the gradient on its logit equals its predicted probability, which is positive at every step and has no opposing signal. Two results support this account. Adding drift corrections to the correction changed nothing measurable (FedProx +0.02 points, p = 0.87; real SCAFFOLD control variates +0.25, +0.43 and +0.38 points across three runs, all non-significant), consistent with the damage sitting in the output layer rather than in the shared feature extractor. With a frozen pretrained backbone the benefit shrank to about 2 points, which bounds the regime in which the correction matters.

## 4. Fractal-chaos guided federated learning

When each hospital computes the descriptors locally and the federated model learns from them, heterogeneity costs 1.3 points at the thesis setting against 15.2 for a CNN on images. The gap of 13.9 points has a 95% confidence interval of [10.8, 17.1] over test patients, so it does not depend on the particular patients tested (decision rule met, Stage 20).

Damage from heterogeneity (centralised minus federated balanced accuracy, points; 20 seeds; same hospitals and test patients). CNN and 19 descriptors: Stage 20; fractal-chaos family alone: Stage 18B.

| Heterogeneity | CNN on images | Fractal-chaos family alone | 19 descriptors |
| --- | --- | --- | --- |
| Severe (alpha 0.1) | 24.1 | 8.2 | 2.9 |
| Moderate (alpha 0.3) | 15.2 | 5.3 | 1.3 |
| Mild (alpha 1.0) | 8.1 | 4.7 | 1.9 |

The gap widens as heterogeneity increases, from 6.2 points [2.5, 9.9] at alpha 1.0 to 21.2 points [18.1, 24.6] at alpha 0.1. The design matched everything except the representation: the same recipe for centralised and federated training, the same hospitals and the same test patients. The CNN reproduced Stage 18 exactly (0.736 centralised, 0.584 federated at alpha 0.3).

Relative to the accuracy each model has to lose, the difference is larger still:

| Model | Centralised accuracy | Relative damage at alpha 0.3 |
| --- | --- | --- |
| CNN on images | 0.736 | 0.377 |
| Fractal-chaos family alone (Stage 18B) | 0.725 | 0.136 |
| 19 descriptors (Stage 20) | 0.869 | 0.024 |

Relative damage is the damage divided by the accuracy above chance. At nearly the same centralised accuracy as the CNN, the fractal-chaos family alone loses about a third as much, and met its decision rule (Stage 18B); the other descriptors resist heterogeneity at least as well, which is recorded in the negative results.

## 5. Methodological results

The thesis measured the sources of inflated accuracy instead of assuming them away, and each measurement narrowed a claim to what the data support.

| Finding | Value | Evidence | Stage |
| --- | --- | --- | --- |
| Inflation from slice-level splitting, hand-crafted descriptors | +3.38 ± 0.79 points, higher in 10 of 10 splits, against 29 to 55 points reported for deep CNNs | Repeated | 10B |
| Gain from majority voting alone (slice to patient) | +6.43 points; slice and patient figures are never compared | Repeated | 10B |
| Per-patient personalisation protocol | An image-blind predictor scores 1.000, so the protocol leaks the label and is not reported | Repeated | Final pipeline |
| Head-mask geometry alone predicts tumour type | 0.750 (0.745 after a normalisation the tumour cannot influence) | Replication of Wallis and Buvat (2022) | 14, 17B |
| Accuracy kept inside strata of equal shortcut information | Tumour-region descriptors 0.97 [0.87, 1.06]; head-region descriptors 0.65; shortcut models 0.68 to 0.74 | Repeated | 15 |
| Shortcut variables added to the descriptor model | +2.1 points, a mechanism by which whole-slice models gain accuracy unrelated to the tumour | Repeated | 15 |
| BraTS: grade information in the descriptors | AUC 0.920; balanced accuracy 0.860 ± 0.043 per fold with a threshold chosen inside training folds | Repeated | 17, 20 |
| BraTS: single fractal-chaos descriptors | C 0.691, SampEn2D 0.687, Ft 0.678, with 95% CIs over patients excluding 0.5 | Repeated | 17 |
| BraTS: shortcut control | Brain geometry at chance (0.464): no Figshare-type shortcut | Repeated | 20 |

In Stage 15 the tumour-region set still contained descriptors that use pixels outside the tumour or encode its size; Stage 20 isolated their share (Section 1). The results are reproducible: Stage 20 reproduced the Stage 19 descriptor-set accuracies and the Stage 18 CNN exactly, and Stage 18B reproduced Stage 18 to the third decimal on different hardware. The only non-deterministic figure was the post-segmentation result of Stage 19 (0.966 and 0.962 in two runs); Stage 20 fixed the GPU settings for deterministic training.

## 6. Where each result is documented

Every figure above can be reproduced from this repository; the committee guide ([`COMMITTEE_GUIDE.md`](../COMMITTEE_GUIDE.md)) gives the order of execution, hardware and run times.

| Section | Scripts (folder `code/`) | Result tables (`results/tables/`) |
| --- | --- | --- |
| 1. Fractal-chaos descriptors | `fractal_chaos_guided_fl/stage20_statistical_corrections.py` (Part 1); `pillar1_classification/stage14b_area_matched.py` | `stage20_part1_figshare.csv`, `stage20_part1_tests.csv`, `stage20_part1_nested.csv`, `stage14b_area_matched.csv` |
| 2. Glioma vs non-glioma; without a radiologist | `stage20_statistical_corrections.py` (Part 2); `pillar1_classification/stage16_radiologist_free_tumour_features.py` | `stage20_part2_glioma.csv`, `stage20_part2_dice.csv`, `stage16_radiologist_free.csv` |
| 3. Federated learning on images | `pillar2_federated/stage6_reviewer_proofing.py`, `stage4_brisc_external.py`, `stage5_alpha_sweep.py`, `stage9_frozen_backbone.py`, `stage13_real_scaffold.py`; `pillar1_classification/full_pipeline.py` | `figshare_main_comparison.csv`, `brisc_external_validation.csv`, `alpha_sweep.csv`, `real_scaffold_stage13.csv`, `ablation_with_scaffold_summary.csv` |
| 4. Fractal-chaos guided federated learning | `stage20_statistical_corrections.py` (Part 4); `fractal_chaos_guided_fl/stage18b_fractal_attribution.py`, `stage19_final_extension.py` (Part 4) | `stage20_part4_damage.csv`, `stage20_part4_gap_bootstrap.csv`, `stage18b_fractal_attribution.csv`, `stage19_part4_federated_v1_1.csv` |
| 5. Methodological results | `pillar1_classification/stage10b_repeated_splits_and_leakage.py`, `stage14_fb_geometry_confound.py`, `stage15_wallis_buvat_shortcut_reliance.py`; `fractal_chaos_guided_fl/stage17_fractal_chaos_descriptors.py`, `stage17b_same_split_threshold_mask_checks.py`, `stage20_statistical_corrections.py` (Part 3) | `leakage_measurement_stage10b.csv`, `stage14_fb_geometry.csv`, `stage15_shortcut_reliance.csv`, `stage17_brats_grading_default_threshold.csv`, `stage17_brats_single_descriptors.csv`, `stage17b_mask_normalisation_check.csv`, `stage20_part3_brats.csv`, `stage20_part3_tests.csv` |
