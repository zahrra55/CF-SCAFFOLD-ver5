# Guide for the Examining Committee

**Thesis:** Fractal-Chaos Guided Federated Learning for Brain Tumor MRI Classification
**Candidate:** Zahraa Ibrahim Khaleel. **Supervisor:** Prof. Dr. Mohammed Sahib Mehdi Al-Taei
**Department of Computer Science, College of Science, Al-Nahrain University**

This guide links every main result of the thesis to the script that produces it, states the hardware
and time each script needs, and lists the limits that apply to each claim. Every decision rule cited
below was written into the script before it was run, and results that did not support a claim are
reported in [`docs/03_withdrawn_and_null.md`](docs/03_withdrawn_and_null.md).

---

## 1. Claims and where their evidence comes from

| Claim | Evidence | Script |
|---|---|---|
| Fractal-chaos descriptors carry substantial class information | Family alone 0.851; removing it lowers the model from 0.928 to 0.785 | `code/fractal_chaos_guided_fl/stage19_final_extension.py` (Part 1), `stage17_fractal_chaos_descriptors.py` |
| Interpretable classification of the three tumour types | 0.948 with 21 descriptors; 0.877 with tumour-only descriptors | `stage19_final_extension.py` (Part 1) |
| Glioma vs non-glioma after automatic segmentation (proxy for malignant vs predominantly benign) | About 0.96 with U-Net masks; 0.778 for the geometry control | `stage19_final_extension.py` (Part 2) |
| Federated learning guided by the descriptors resists heterogeneity | Damage 1.7 points (descriptors) vs 15.2 (CNN on images); federated 0.900, patient level 0.953 | `stage18_matched_heterogeneity_damage.py`, `stage19_final_extension.py` (Part 4) |
| The class-presence correction repairs federated training on images | 0.649 vs 0.584 (FedAvg); equivalent to logit calibration; no collapse in 20 seeds | `code/pillar2_federated/stage6_reviewer_proofing.py`, `code/pillar1_classification/full_pipeline.py` |
| Classification is possible without a radiologist | No detectable cost with the full model (-0.6 points, 95% CI -4.6 to +3.7) | `code/pillar1_classification/stage16_radiologist_free_tumour_features.py` |
| The dataset shortcut is measured, not ignored | Brain geometry alone 0.750; tumour-region features retain 0.97 of their accuracy inside shortcut strata | `stage14_fb_geometry_confound.py`, `stage15_wallis_buvat_shortcut_reliance.py` |

## 2. Environment

All experiments ran on the Kaggle platform with Python 3, PyTorch and scikit-learn; the packages are
listed in [`requirements.txt`](requirements.txt). Each script installs `h5py`, `scikit-image` and
`nibabel` if they are missing. Each script is self-contained: open it in a Kaggle notebook cell, attach
the datasets of Section 3, and run it. Results print to the notebook and are written to
`/kaggle/working`; long scripts cache their features and resume after an interrupted session.

## 3. Datasets (attach as Kaggle datasets; none is redistributed here)

| Dataset | Use | Source |
|---|---|---|
| Figshare brain tumour dataset (Cheng et al., 2015): 3064 T1-CE slices, 233 patients, tumour masks, patient IDs | All descriptor and federated experiments | doi:10.6084/m9.figshare.1512427 |
| BraTS 2020 training set, with `name_mapping.csv` (grade labels) | High-grade vs low-grade glioma | BraTS 2020 challenge data (Kaggle mirrors titled "BraTS2020") |
| BRISC (Fateh et al., 2026) | External validation of the federated correction | doi:10.1038/s41597-026-06753-y |

## 4. Order of execution, hardware and time

The final stages can be run on their own; the earlier stages document how the method was reached.

| Order | Script | Hardware | Time (measured) | Produces |
|---|---|---|---|---|
| 1 | `code/pillar1_classification/full_pipeline.py` | GPU | about 2 h | Corrected 17-descriptor model, U-Net delineation, federated ablation with SCAFFOLD, leakage guard |
| 2 | `code/pillar1_classification/stage14_fb_geometry_confound.py`, `stage14b_area_matched.py`, `stage15_wallis_buvat_shortcut_reliance.py` | CPU | 20 to 30 min each | Shortcut analyses |
| 3 | `code/pillar1_classification/stage16_radiologist_free_tumour_features.py` | GPU | about 15 min | Cost of removing the radiologist |
| 4 | `code/fractal_chaos_guided_fl/stage17_fractal_chaos_descriptors.py` | CPU | about 35 min | SampEn2D, descriptor-based federated learning, BraTS grading |
| 5 | `code/fractal_chaos_guided_fl/stage17b_same_split_threshold_mask_checks.py` | GPU | about 50 min | Same-split test, grading threshold, mask check |
| 6 | `code/fractal_chaos_guided_fl/stage18_matched_heterogeneity_damage.py` | GPU | about 85 min | Matched heterogeneity damage |
| 7 | `code/fractal_chaos_guided_fl/stage18b_fractal_attribution.py` | CPU | about 40 min | Attribution to the fractal-chaos family |
| 8 | `code/fractal_chaos_guided_fl/stage19_final_extension.py` (version 1.1) | GPU | about 2 h 45 min | Final descriptor set, glioma vs non-glioma, BraTS sub-regions, federated arm |

The federated experiments on images of Part II (`code/pillar2_federated/`) are described stage by stage
in [`docs/02_pillar2_federated.md`](docs/02_pillar2_federated.md) and mapped to their tables in
[`code/README.md`](code/README.md).

## 5. Reproducibility

Splits, partitions, random forests and network initialisations are seeded. Descriptor-based results are
deterministic: Stage 19 reproduced Parts 1 and 3 exactly across two runs, and Stage 18B reproduced Stage 18
to the third decimal on different hardware. Training a U-Net on a GPU is not fully deterministic; the
post-segmentation figure of Stage 19 was 0.966 and 0.962 in two runs, so it is reported as about 0.96.

## 6. Limits that apply to the claims

1. **Glioma vs non-glioma is a proxy.** Meningiomas and pituitary adenomas are mostly benign and gliomas
   malignant, but atypical meningiomas and low-grade gliomas exist. BraTS low-grade gliomas are WHO
   grade 2 and malignant, so the BraTS task is grading, not benign vs malignant.
2. **The final 0.948 is slightly optimistic,** because the descriptor set was selected on the same data
   by a preregistered rule; the tumour-only 0.877 is the shortcut-robust estimate.
3. **Descriptors use a tumour mask.** The federated comparison with the CNN uses radiologist masks; the
   cost of automatic masks was measured centrally (Stage 16, Stage 19 Part 2) but not in the federated arm.
4. **Robustness is not specific to the fractal-chaos family.** The non-fractal descriptors resist
   heterogeneity at least as well (Stage 18B).
5. **BraTS grading.** The fractal measures of tumour sub-regions added nothing beyond the presence of
   enhancement and necrosis, and low-grade cases come mostly from one collection (TCGA-LGG), so grade is
   partly confounded with acquisition site.
6. **One primary dataset.** All descriptor results come from the Figshare collection; BRISC derives
   partly from the same public sources.
