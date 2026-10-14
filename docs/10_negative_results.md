# CF-SCAFFOLD: Negative Results, Bounded Claims and Comparisons

October 2026. Companion file: [`09_positive_results.md`](09_positive_results.md).

## How to read this file

This file is the companion of *CF-SCAFFOLD: Positive Results*. It records every result that did not support a claim, every claim that was withdrawn, every method rejected, every error found by an audit, and the comparisons that bound each claim. They are recorded because they shaped the final method, and because a reader checking the thesis should find them rather than a gap.

| Category | Meaning |
| --- | --- |
| Not confirmed | A decision rule fixed before the run was not met |
| Withdrawn | Reported or believed at one point, then shown to be wrong or overstated |
| Null | Tested properly; no measurable effect |
| Rejected | A method dropped before running or during validation |
| Corrected | An error found by an audit and fixed |
| Bounding comparison | A result or limit that restricts how far a claim can be stated |

Where the corrected statistics of Stage 20 re-tested a result, their figures are the ones that count. p-values from Wilcoxon tests over repeated cross-validation splits (Stages 10B to 19) are not valid as stated, because the splits share patients; they are quoted only where they mark the history of a decision. Tests over training seeds on a fixed test set are conditional on those test patients.

## Summary: the negative results that changed the thesis

The most consequential negative result was methodological: the statistical test used from Stage 10B to Stage 19 overstated significance, and correcting it reversed two decisions.

| Result | Category | What it changed |
| --- | --- | --- |
| Wilcoxon tests over repeated cross-validation splits overstate significance | Corrected | All descriptor comparisons re-tested with the corrected t-test (Stage 20) |
| Grayscale lacunarity: +2.38 points [-0.45, +5.20], p = 0.098 | Not confirmed | Final set stays at 19 descriptors: 0.923, not 0.948 |
| FuzEn2D non-inferior to SampEn2D: CI lower bound -1.90 vs required -0.50 | Not confirmed | SampEn2D kept |
| Strictly tumour-only descriptors: 0.736, 9.1 points below the earlier tumour-only set | Bounding comparison | The shortcut-robust estimate is 0.736, not 0.877 |
| Glioma vs non-glioma margin over the shortcut control: lower bound 0.091 vs required 0.10 | Not confirmed | The margin is stated without the 10-point claim |
| Robustness to heterogeneity is not specific to the fractal-chaos family | Bounding comparison | "Guided" is worded as descriptors used as inputs, not as a unique property of fractal-chaos |
| The class-presence correction lowers accuracy when federated learning uses descriptors | Not confirmed | The correction is presented as a method for image models only |
| C = Ft/Fb adds nothing to Ft and Fb | Null | C is presented as an interpretable member of the fractal family, not a superior descriptor |
| BraTS: fractal measures of tumour sub-regions add nothing | Not confirmed | The sub-region grade signal is attributed to the presence of enhancement and NCR/NET |
| Clustered federated learning (+12.5 points) and "exceeds Cheng et al. (2015)" | Withdrawn | Removed from all claims |

## 1. Not confirmed by the final corrected statistics (Stage 20)

Stage 20 re-tested every descriptor comparison with the corrected repeated cross-validation t-test and every federated or segmentation comparison with a bootstrap over patients. Five comparisons did not meet their rules, and one earlier conclusion was withdrawn.

| Comparison | Mean difference [95% CI], corrected p | Rule (fixed before the run) | Outcome |
| --- | --- | --- | --- |
| Grayscale lacunarity added to the 19 descriptors | +2.38 points [-0.45, +5.20], p = 0.098 (Holm 0.196) | Positive gain, Holm p < 0.05 | Not accepted |
| Boundary fractal dimension added to the 19 descriptors | -0.35 points [-2.13, +1.43], p = 0.695 | Positive gain, Holm p < 0.05 | Not accepted |
| FuzEn2D instead of SampEn2D | +0.30 points [-1.90, +2.50], p = 0.785 | Defined on as many slices, and CI lower bound >= -0.50 points | Not adopted (coverage met: 0 vs 31 undefined) |
| Glioma vs non-glioma: U-Net (intention-to-treat) minus geometry control | +0.149 [0.091, 0.209] | Accuracy lower bound >= 0.85 (met: 0.915) and margin lower bound >= 0.10 | Not supported |
| BraTS: sub-region block added to the 19 descriptors | +0.49 points [-6.01, +7.00], p = 0.879 | Positive gain, p < 0.05 | Not accepted |
| BraTS: sub-region fractal measures vs presence flags | -2.44 points [-6.32, +1.44], p = 0.212 | Descriptive | The Stage 19 conclusion that the flags carry more (Wilcoxon p = 0.002) is withdrawn |

Three notes bound how these outcomes should be read. First, the non-inferiority margin of 0.5 points was unreachable by design: with ten repeats of five folds, the confidence intervals of this test extend 1.8 to 2.8 points on each side of the mean. The rule was fixed before the run and stands; the error lay in choosing the margin. Second, grayscale lacunarity was included in the set chosen in all 15 outer folds of nested selection, whose estimate (0.939) exceeded the rule-chosen set on the same folds (0.918); its benefit is probable but is not confirmed by a single test. Third, in the glioma comparison the accuracy condition was met and the margin is clearly above zero; only the 10-point threshold was not confirmed at the 95% level.

## 2. Claims withdrawn and figures superseded

Two claims were withdrawn after proper testing, one result was invalid by construction, and several figures and labels were superseded as the evaluation became stricter.

| Claim or figure | Original | What testing showed | Action |
| --- | --- | --- | --- |
| Clustered federated learning (clients grouped by their C distribution) | +12.5 points, one seed | Ten seeds, capacity-matched: Holm p 0.08 to 0.10; cluster stability ARI 0.017; stabilised, 20 seeds: +0.012, p = 0.88, 10 of 20 wins; the clustered model had degenerated to about 0.505 | Withdrawn |
| Per-patient personalisation | +32 points, d = 10.38, no patient harmed | Each patient has one tumour type, so the support labels give away the query labels: an image-blind predictor scores 1.000 | Discarded; the leakage check was kept in the pipeline |
| Descriptor model exceeds Cheng et al. (2015) | 0.919 vs 0.913 | After the feature-code corrections, 0.897 | Withdrawn, and not reinstated for 0.923 or 0.939 because metric (balanced vs plain accuracy), level (patient vs slice) and shortcut exposure differ |
| C adds about 1.3 points | One split (Stage 10) | Ten splits: -0.08 points, 3 of 10 | Withdrawn as split-to-split noise |
| Cost of removing the radiologist: 0.2 points | Final pipeline | Bootstrap over patients: -0.6 points, CI -4.6 to +3.7 | Restated as no detectable cost, with an uncertainty of about four points |
| Stage 19 figures: 0.948 (21 descriptors), 0.877 (tumour-only), 0.851 (family), federated 0.900 / 0.953 | Stage 19 | They depended on accepting grayscale lacunarity and FuzEn2D, which Stage 20 did not confirm | Superseded by 0.923, 0.736 (strict), 0.756 and 0.856 / 0.923 |
| BraTS: presence flags carry more than the sub-region fractal measures | Wilcoxon p = 0.002 (Stage 19) | Corrected p = 0.212 | Withdrawn |
| Patient-level federated figures of Stage 17 and Stage 19 version 1.0 | Ties broken towards the lowest class | Corrected in Stage 19 version 1.1 | Superseded |
| "Preregistered" | Stages 13 to 19 | Rules were fixed before each run but not registered externally, and the same data had been analysed before | Relabelled "decision rules fixed before the run" |
| Fb: fractal dimension of the brain excluding the tumour | Stages 9 to 19 | Figshare images are not skull-stripped, so the mask covers the head | Renamed: head region excluding the tumour (on BraTS it remains the brain) |
| BraTS label 1: necrosis | Stage 19 | In BraTS 2017 to 2020, label 1 is necrotic and non-enhancing tumour core (NCR/NET) | Renamed NCR/NET |

## 3. Null results and hypotheses not supported

Eleven hypotheses were tested and not supported. Several of these nulls are informative: three independent drift corrections changing nothing is part of the evidence that the damage sits in the output layer.

| Hypothesis | Test | Result | Reading |
| --- | --- | --- | --- |
| A bio-anchored contrastive term adds at least 2 points to the correction | 16 seeds; rule fixed before the run | +1.26 points (11 of 16 seeds) | Criterion not met; without masking, the contrastive term was 10 points worse |
| Weighting aggregation by C counteracts drift | Against SCAFFOLD | +0.012, p = 0.812 | Null, as expected: control variates already target that drift |
| FedProx adds to the correction | 20 seeds | +0.02 points, p = 0.87 | Null |
| Real SCAFFOLD control variates add to the correction | Three runs of 20 seeds | +0.25 (p = 0.546), +0.43 (p = 0.230), +0.38 points (p = 0.43) | Null in all three; consistent with damage in the output layer |
| A frozen pretrained backbone amplifies the gain of the correction | Figshare and BRISC | Gain shrinks to +2.2 and +1.9 points; corrections become indistinguishable | Prediction failed; anticipated by the pre-training literature |
| C adds to Ft and Fb | Ten splits | -0.08 points (3 of 10); C instead of Ft and Fb: -1.19 points (0 of 10) | Null; dividing discards the scale of both dimensions |
| The class-presence correction helps federated learning on descriptors | 20 seeds (Stage 17) | -2.5 points, 0 of 20 seeds | Hypothesis failed: with fixed descriptors, the absent-class signal is useful |
| BraTS: the fractal-chaos family reaches 0.70, at least 0.05 above the geometry control | Stage 17 | 0.646 vs 0.485 (0.702 with an in-fold threshold, exploratory) | Not supported |
| First matched test: CNN damage exceeds descriptor damage by at least 5 points | Stage 17B | 6.6 vs 3.5 points (difference 3.1) | Not supported; two design flaws, corrected in Stages 18 and 20 |
| Robustness to heterogeneity is specific to the fractal-chaos family | Stage 18B | Other 12 descriptors: -0.1 points; family: 5.3 points | Not specific: fixed descriptors in general resist heterogeneity |
| Tumour-region descriptors without a radiologist cost at most 2 points | Stage 16 rule: >= -2 holds, <= -5 fails | -3.8 points [-7.9, 0.0] | Partial; shape descriptors agree poorly between masks (r 0.25 to 0.45) |

The p-values of the C comparisons and of the Stage 18B attribution come from Wilcoxon tests over splits or seeds and are not valid as stated; the magnitudes and the counts of splits or seeds are the evidence.

## 4. Methods rejected before running or during validation

Every new descriptor and method was first tested on cases with a known answer; those that failed never reached the real data. Two proposals were declined because they would have chosen the design after seeing the results.

| Method | Where | Why it was rejected |
| --- | --- | --- |
| Automatic segmentation by active contour (Chan-Vese) | Stage 11 | Mean Dice 0.034, median 0; it worked only when started from the expert centroid (Dice 0.656), so locating the tumour was the problem; replaced by a U-Net localiser |
| Active-contour refinement of the U-Net region | Final pipeline | Lowered Dice on real images (-0.047), although it had helped on synthetic data |
| Bilateral-symmetry seeding of the contour | Stage 12 | Worked on synthetic data only (Dice 0.658 and 0.622); never run on real data, so no claim is made |
| A random-mask control | Before running | With three classes it cannot avoid masking classes a client holds |
| SampEn2D with template length m = 2 | Stage 17 validation | Undefined on white noise; m = 1 passed every test |
| Roughness exponent and Higuchi dimension of the boundary signature | Stage 19 validation | Dominated by pixel quantisation: for the same smooth disc the roughness exponent was 0.95 at one size and 0.37 at another |
| Subtracting a baseline before grayscale lacunarity | Stage 19 validation | Reversed the ordering of homogeneous and gappy textures |
| Lyapunov exponents, correlation dimension, recurrence quantification | Planning | Require a time series; not meaningful on a static image region |
| Multifractal spectra (2D MF-DFA), permutation and dispersion entropy | Planning | Too few boxes or patterns on small tumours; complexity without a reliable expected gain |
| A non-linear projection head added after the contrastive null | Declined | Changing the design after seeing the result and testing against the same threshold is not valid |
| Removing shape descriptors after Stage 16 | Declined | Choosing descriptors after seeing results on the same data; left as a hypothesis for future work |

## 5. Errors found by audits and corrected

Errors were found by five checks: a feature-code audit before the final pipeline, an audit of a colleague's script (Stage 12), testing during Stage 17, an audit of Stage 19 before submission, and an external code review that led to Stage 20. Each error was verified before it was fixed, most with a test that the old code fails.

| Error | Found by | Effect | Fix |
| --- | --- | --- | --- |
| F1. Entropy computed from a density instead of probabilities | Feature audit | Entropy depended on the intensity range, not on the shape of the distribution | Probabilities |
| F2. GLCM computed on the whole image with the background set to zero | Feature audit | Texture mainly measured tumour size and border | Inside the region only, 1st to 99th percentile quantisation |
| F3. DBC box height h = s instead of s · G / M | Feature audit | Biased fractal dimensions | h = s · G / M (Sarkar and Chaudhuri, 1994) |
| F4. Fractal dimension of the tumour missing | Feature audit | Incomplete fractal family | Ft added (17 descriptors) |
| F5. Patient-vote ties broken by class order | Feature audit | Biased patient-level accuracy | Ties broken by mean probability |
| F6. SCAFFOLD global variate sample-weighted and scaled by 17/18 | Feature audit | Deviation from SCAFFOLD Option II | Plain mean |
| F7. Documentation said the seed also changed the test split | Feature audit | Misreadable protocol | Documented |
| F8. Lacunarity skipped the last row and column | Feature audit | Small bias | Counted |
| A colleague's DBC script: intensity wrap-around in uint8, normalisation over a zero-filled array, thresholds for 128 px applied at 512 px | Stage 12 audit | 3064 of 3065 slices kept; meningioma AUC 0.467 | Corrected copy kept as an audit record only |
| BraTS brain mask computed after slice normalisation | Stage 17 testing | The tumour shifted the mask; the geometry control scored 1.000 on synthetic slices with identical brains | Mask from raw intensities (control 0.500) |
| Federated patient votes received one-hot predictions | Stage 19 audit | Ties went to the lowest class (patient-level 0.951 instead of 0.953) | Model probabilities |
| Federated imputation by a median of pooled data | Stage 19 audit | Not computable in a federation | Mean from aggregated sums and counts |
| Wilcoxon tests over repeated cross-validation splits | External review | p-values too small (Stages 10B to 19) | Corrected repeated cross-validation t-test (Stage 20) |
| Final set selected and evaluated on the same folds | External review | Optimistic estimate | Nested selection (Stage 20) |
| Blocks tested on the SampEn2D base while the final set used FuzEn2D | External review | Combination never tested | FuzEn2D decided first, blocks tested on the chosen base |
| Segmentation arm: failures dropped, Dice computed without failures, control on another split | External review | Optimistic post-segmentation figure | Intention-to-treat, Dice over all slices, the same split for every arm |
| Part 2 labels wrong when Part 1 is disabled (B1) | External review | Mislabelled output (did not affect the runs reported) | Sets derived from the saved final set |
| Caches without a version check (B3); hand-set final set (B4) | External review | Risk of silently wrong columns or sets | Version signatures; final set saved by Part 1 |
| Segmentation files matched case-sensitively (B5) | External review | One BraTS case dropped (368 of 369) | Case-insensitive matching; all 369 used |
| Clients dropped by the partition not reported (B7) | External review | Hidden difference in training data | Reported: mean 0.3, at most 5 slices |
| Claimed: the two segmentation arms used different image quantisation (B2) | External review | Tested: the images are identical (maximum difference 0) | Not an error; no change |

## 6. Comparisons and limits that bound each claim

Each positive claim holds only within the limits below, and the thesis states it within them.

| Claim | Bounding comparison or limit | How the claim is stated |
| --- | --- | --- |
| Fractal-chaos descriptors carry class information | The family includes head-region descriptors (Fb, C, head lacunarity), which partly follow the slice-selection shortcut; Ft correlates with tumour size (rho +0.65) and tumour lacunarity almost entirely encodes size (rho -0.98) | The contribution (+13.2 points) is reported together with the strict tumour-only estimate (0.736) |
| Three-class classification | Strictly tumour-only descriptors fall 9.1 points below the earlier tumour-only set [-14.4, -3.9], p = 0.001; head geometry alone reaches 0.750; all shortcut variables together (0.857) beat tumour-region descriptors (0.809) | 0.923 is stated with 0.736 as the shortcut-robust estimate |
| Comparison with published accuracies | Cheng et al. (2015) report 0.913 plain accuracy scored per image; figures here are balanced accuracies scored per patient, and partly shortcut-exposed | No claim to exceed published results |
| Descriptor-guided federated learning resists heterogeneity | Descriptors use the radiologist's mask (the cost of automatic masks was measured centrally, not in the federated arm); fixed non-fractal descriptors resist as well | Stated as a property of fixed, interpretable descriptors used as inputs |
| The class-presence correction repairs federated training on images | Equivalent to logit calibration (0.649 vs 0.653, Holm p = 0.123); below centralised training (0.705); about +2 points with a frozen backbone; +0.2 points at low skew; tests over seeds are conditional on the test patients | Stated as matching the strongest published correction without a tuned parameter, in the regime of strong skew and a trainable backbone |
| Glioma vs non-glioma | A proxy: atypical meningiomas and low-grade gliomas exist; the 10-point margin over the shortcut control was not confirmed | Stated as glioma vs non-glioma, with the margin and its interval |
| BraTS grading | Grade, not benign vs malignant (low-grade gliomas are WHO grade 2 and malignant); low-grade cases come mostly from TCGA-LGG; sub-regions and slice selection use the expert segmentation; the tumour mask includes oedema | Stated as an upper bound under site confounding |
| Leakage from slice-level splitting | 3.38 points here against 29 to 55 points reported for deep CNNs | Stated as dependent on model capacity |
| Novelty | The shortcut is a replication of Wallis and Buvat (2022); the frozen-backbone observation was anticipated (Nguyen et al., 2023; Chen et al., 2023; Park et al., 2025); class-presence masking is the alpha = 0 case of FedRS | These are presented as replications or bounded findings, not contributions |
| Generalisation | One primary dataset; external validation (BRISC) only for the federated correction on images, and BRISC derives partly from the same public sources | Stated as a limitation |
| Title wording | Sample and fuzzy entropy are complexity measures from the chaos-theory tradition, not measures of chaos such as Lyapunov exponents; "guided" means the federated model learns from the descriptors | Both terms are defined in Chapter 1 |

## 7. Where each item is documented

| Section | Documentation | Scripts (folder `code/`) | Result tables (`results/tables/`) |
| --- | --- | --- | --- |
| 1. Not confirmed (Stage 20) | [`08_later_experiments.md`](08_later_experiments.md) | `fractal_chaos_guided_fl/stage20_statistical_corrections.py` | `stage20_part1_tests.csv`, `stage20_part1_figshare.csv`, `stage20_part2_glioma.csv`, `stage20_part3_tests.csv` |
| 2. Withdrawn and superseded | [`03_withdrawn_and_null.md`](03_withdrawn_and_null.md), [`07_verification_log.md`](07_verification_log.md) | `shared/leakage_guard.py`, `pillar1_classification/full_pipeline.py`, `pillar1_classification/stage16_radiologist_free_tumour_features.py` | `stage16_radiologist_free.csv` |
| 3. Null results | [`03_withdrawn_and_null.md`](03_withdrawn_and_null.md), [`08_later_experiments.md`](08_later_experiments.md) | `pillar2_federated/stage7_bio_contrastive.py`, `stage9_frozen_backbone.py`, `stage9b_frozen_backbone_brisc.py`, `stage13_real_scaffold.py`; `pillar1_classification/stage10b_repeated_splits_and_leakage.py`; `fractal_chaos_guided_fl/stage17_fractal_chaos_descriptors.py`, `stage17b_same_split_threshold_mask_checks.py`, `stage18b_fractal_attribution.py` | `bio_contrastive_null.csv`, `real_scaffold_stage13.csv`, `descriptor_ablation_stage10b.csv`, `stage17_fractal_chaos_figshare.csv`, `stage17_brats_grading_default_threshold.csv`, `stage18b_fractal_attribution.csv` |
| 4. Rejected methods | [`08_later_experiments.md`](08_later_experiments.md) | `pillar1_classification/stage11_active_contour.py`, `stage12_dbc_audit_and_symmetry_seed.py`; `fractal_chaos_guided_fl/stage19_final_extension.py` | |
| 5. Errors corrected | [`07_verification_log.md`](07_verification_log.md) | `pillar1_classification/full_pipeline.py` (header lists F1 to F8), `audits/colleague_pipeline_resolution_fix.py`, `fractal_chaos_guided_fl/stage19_final_extension.py` (version 1.1), `stage20_statistical_corrections.py` | |
| 6. Bounding comparisons | [`08_later_experiments.md`](08_later_experiments.md), [`COMMITTEE_GUIDE.md`](../COMMITTEE_GUIDE.md) | `pillar1_classification/stage15_wallis_buvat_shortcut_reliance.py`, `fractal_chaos_guided_fl/stage20_statistical_corrections.py` | `stage15_shortcut_reliance.csv`, `stage20_part1_tests.csv` |
