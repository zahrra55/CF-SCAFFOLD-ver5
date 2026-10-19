# Guide for the Examining Committee

**Thesis:** Fractal-Chaos Guided Federated Learning for Brain Tumor MRI Classification
**Candidate:** Zahraa Ibrahim Khaleel. **Supervisor:** Prof. Dr. Mohammed Sahib Mehdi Al-Taei
**Department of Computer Science, College of Science, Al-Nahrain University**

This guide links every main result of the thesis to the script that produces it, states the hardware
and time each script needs, and lists the limits that apply to each claim. The final figures come from
Stage 20, which corrected the statistics after an external code review: descriptor comparisons use the
corrected repeated cross-validation t-test (Nadeau and Bengio, 2003) and federated comparisons a bootstrap
over test patients. Decision rules were written into each script before it was run; they were not
registered externally, and the same data had been analysed in earlier stages. Every positive result is
listed in [`docs/09_positive_results.md`](docs/09_positive_results.md), and every result that did not
support a claim in [`docs/10_negative_results.md`](docs/10_negative_results.md).

---

## 1. Claims and where their evidence comes from

| Claim | Evidence | Script |
|---|---|---|
| Fractal-chaos descriptors carry substantial class information | Removing the family from the 19 descriptors costs 13.2 points (95% CI 7.2 to 19.1, corrected p < 0.0001) | `code/fractal_chaos_guided_fl/stage20_statistical_corrections.py` (Part 1) |
| Interpretable classification of the three tumour types | 0.923 with 19 descriptors; 0.939 by nested selection; 0.736 with strictly tumour-only descriptors | `stage20_statistical_corrections.py` (Part 1) |
| Glioma vs non-glioma after automatic segmentation (proxy for malignant vs predominantly benign) | 0.946 (95% CI 0.915 to 0.973) under intention-to-treat; 0.797 for the geometry control | `stage20_statistical_corrections.py` (Part 2) |
| Federated learning guided by the descriptors resists heterogeneity | Damage 1.3 points (descriptors) vs 15.2 (CNN on images); gap 13.9 points, 95% CI 10.8 to 17.1 over test patients | `stage20_statistical_corrections.py` (Part 4) |
| The class-presence correction repairs federated training on images | 0.649 vs 0.584 (FedAvg); equivalent to logit calibration; no collapse in 20 seeds | `code/pillar2_federated/stage6_reviewer_proofing.py`, `code/pillar1_classification/full_pipeline.py` |
| Classification is possible without a radiologist | No detectable cost with the full model (-0.6 points, 95% CI -4.6 to +3.7) | `code/pillar1_classification/stage16_radiologist_free_tumour_features.py` |
| The dataset shortcut is measured, not ignored | Head geometry alone 0.750; tumour-region descriptors retain 0.97 of their accuracy inside shortcut strata | `stage14_fb_geometry_confound.py`, `stage15_wallis_buvat_shortcut_reliance.py` |

## 2. Environment

All experiments ran on the Kaggle platform (GPU T4) with Python 3, PyTorch and scikit-learn; the packages
are listed in [`requirements.txt`](requirements.txt), and Stage 20 prints the exact library versions at
start. Each script installs `h5py`, `scikit-image` and `nibabel` if they are missing. Each script is
self-contained: open it in a Kaggle notebook cell, attach the datasets of Section 3, and run it. Results
print to the notebook and are written to `/kaggle/working`; long scripts cache their work and resume after
an interrupted session (Stage 20 rejects a cache built by different code).

## 3. Datasets (attach as Kaggle datasets; none is redistributed here)

| Dataset | Use | Source |
|---|---|---|
| Figshare brain tumour dataset (Cheng et al., 2015): 3064 T1-CE slices, 233 patients, tumour masks, patient IDs | All descriptor and federated experiments | doi:10.6084/m9.figshare.1512427 |
| BraTS 2020 training set, with `name_mapping.csv` (grade labels) | High-grade vs low-grade glioma | BraTS 2020 challenge data (Kaggle mirrors titled "BraTS2020") |
| BRISC (Fateh et al., 2026) | External validation of the federated correction | doi:10.1038/s41597-026-06753-y |

## 4. Order of execution, hardware and time

**Shortest route to the final results:** the self-contained folder [`FractalChaos-FL-BrainTumor-Lab/`](FractalChaos-FL-BrainTumor-Lab/) reproduces every result in Section 1 with six experiments, run in order (instructions and expected values in its README). The stage scripts below are the full experimental record from which that folder was assembled.

Stage 20 alone reproduces the first four claims of Section 1. The other three (the class-presence
correction on images, classification without a radiologist, and the shortcut measurements) come from the
earlier stages, which also document how the method was reached.

| Order | Script | Hardware | Time (measured) | Produces |
|---|---|---|---|---|
| 1 | `code/pillar1_classification/full_pipeline.py` | GPU | about 2 h | Corrected 17-descriptor model, U-Net delineation, federated ablation with SCAFFOLD, leakage guard |
| 2 | `code/pillar1_classification/stage14_fb_geometry_confound.py`, `stage14b_area_matched.py`, `stage15_wallis_buvat_shortcut_reliance.py` | CPU | 20 to 30 min each | Shortcut analyses |
| 3 | `code/pillar1_classification/stage16_radiologist_free_tumour_features.py` | GPU | about 15 min | Cost of removing the radiologist |
| 4 | `code/fractal_chaos_guided_fl/stage17_fractal_chaos_descriptors.py` | CPU | about 35 min | SampEn2D, descriptor-based federated learning, BraTS grading |
| 5 | `code/fractal_chaos_guided_fl/stage17b_same_split_threshold_mask_checks.py` | GPU | about 50 min | Same-split test, grading threshold, mask check |
| 6 | `code/fractal_chaos_guided_fl/stage18_matched_heterogeneity_damage.py` | GPU | about 85 min | Matched heterogeneity damage |
| 7 | `code/fractal_chaos_guided_fl/stage18b_fractal_attribution.py` | CPU | about 40 min | Attribution to the fractal-chaos family |
| 8 | `code/fractal_chaos_guided_fl/stage19_final_extension.py` (version 1.1) | GPU | about 2 h 45 min | Descriptor extensions, glioma vs non-glioma, BraTS sub-regions, federated arm (statistics superseded by Stage 20) |
| 9 | `code/fractal_chaos_guided_fl/stage20_statistical_corrections.py` | GPU | about 4 h 10 min | **Final figures:** corrected tests, nested selection, strict tumour-only set, glioma vs non-glioma under intention-to-treat, BraTS with all 369 cases, Stage 18 rerun with a bootstrap over test patients |

The federated experiments on images of Part II (`code/pillar2_federated/`) are described stage by stage
in [`docs/02_pillar2_federated.md`](docs/02_pillar2_federated.md) and mapped to their tables in
[`code/README.md`](code/README.md).

## 5. Reproducibility

Splits, partitions, random forests and network initialisations are seeded. Descriptor-based results are
deterministic: Stage 20 reproduced the Stage 19 descriptor-set accuracies and the Stage 18 CNN exactly, and
Stage 18B reproduced Stage 18 to the third decimal on different hardware. U-Net training on a GPU was not
deterministic in Stage 19 (0.966 and 0.962 in two runs); Stage 20 fixes the cuDNN settings for
deterministic training.

## 6. Limits that apply to the claims

1. **Glioma vs non-glioma is a proxy.** Meningiomas and pituitary adenomas are mostly benign and gliomas
   malignant, but atypical meningiomas and low-grade gliomas exist. The margin of 0.149 over the shortcut
   control is above zero, but its lower bound (0.091) missed the 0.10 required by the decision rule.
2. **The shortcut-robust estimate is 0.736,** from strictly tumour-only descriptors. The descriptor family
   includes head-region descriptors (Figshare is not skull-stripped, so the "brain" mask covers the head),
   and some tumour descriptors mostly encode tumour size.
3. **Two extensions were not confirmed.** Grayscale lacunarity (+2.38 points, p = 0.098) and FuzEn2D
   (non-inferiority not established) did not pass the corrected test; the final set is the 19 descriptors.
4. **Descriptors use a tumour mask.** The federated comparison with the CNN uses radiologist masks; the
   cost of automatic masks was measured centrally (Stage 16, Stage 20 Part 2) but not in the federated arm.
5. **Robustness is not specific to the fractal-chaos family.** The non-fractal descriptors resist
   heterogeneity at least as well (Stage 18B).
6. **BraTS grading.** Grade, not benign vs malignant (low-grade gliomas are WHO grade 2 and malignant). The
   fractal measures of tumour sub-regions added nothing (p = 0.879); low-grade cases come mostly from one
   collection (TCGA-LGG), so grade is partly confounded with acquisition site; sub-regions and slice
   selection use the expert segmentation, and label 1 is the necrotic and non-enhancing core (NCR/NET).
7. **One primary dataset.** All descriptor results come from the Figshare collection; BRISC derives
   partly from the same public sources.
