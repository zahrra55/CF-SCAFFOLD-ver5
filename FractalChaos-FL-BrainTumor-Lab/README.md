# Fractal-Chaos Guided Federated Learning for Brain Tumour MRI Classification

Code for the final results of the M.Sc. thesis *Fractal-Chaos Guided Federated Learning for Brain Tumor MRI
Classification* (Zahraa Ibrahim Khaleel; supervisor Prof. Dr. Mohammed Sahib Mehdi Al-Taei; Department of
Computer Science, College of Science, Al-Nahrain University).

The folder is self-contained. It computes fractal, entropy, texture and shape descriptors of brain tumours on
MRI, classifies tumour types and grades from them, and compares federated learning on these descriptors with
federated learning on images when hospitals hold different subsets of tumour classes.

## Contents

```
fcfl/                     library
  config.py               paths and settings
  io.py                   reading Figshare and BraTS data
  descriptors.py          fractal dimension, lacunarity, 2D sample and fuzzy entropy, texture, shape, geometry
  data.py                 building and caching the descriptor datasets; U-Net masks and their descriptors
  segmentation.py         U-Net tumour localiser
  evaluation.py           patient-grouped cross-validation, threshold selection, nested selection
  federated.py            hospital partitions, models, federated training, local objectives
  stats.py                corrected repeated cross-validation t-test, Holm correction, bootstrap intervals
experiments/
  01_descriptor_classification.py   three-class classification; selects the descriptor set
  02_glioma_vs_nonglioma.py         glioma versus non-glioma after automatic segmentation
  03_brats_grading.py               high-grade versus low-grade glioma on BraTS 2020
  04_heterogeneity_damage.py        damage from heterogeneity: CNN on images versus descriptors
  05_fl_local_objectives.py         FedAvg, FedRS, FedLC and class-presence restriction on images
  06_radiologist_free.py            cost of replacing the radiologist's mask by the U-Net mask
```

## Data

| Dataset | Used in | Source |
|---|---|---|
| Figshare brain tumour dataset (Cheng et al., 2015): 3064 contrast-enhanced T1 slices, 233 patients, tumour masks | Experiments 1, 2, 4, 5, 6 | doi:10.6084/m9.figshare.1512427 |
| BraTS 2020 training data with `name_mapping.csv` | Experiment 3 | BraTS 2020 challenge (Kaggle mirrors titled "BraTS2020") |

No data are included in this folder.

## Running on Kaggle

Attach the datasets to a notebook with a GPU (one T4 is enough), then:

```
!git clone https://github.com/zahraa55/CF-SCAFFOLD.git
%cd CF-SCAFFOLD/FractalChaos-FL-BrainTumor-Lab
!pip install -q -r requirements.txt
!python experiments/01_descriptor_classification.py
```

Run the experiments in order: experiment 1 selects the descriptor set used by experiments 2, 4 and 6, and
experiment 6 reuses the U-Net masks of experiment 2. Paths default to `/kaggle/input` and `/kaggle/working`
and can be changed with the environment variables `FCFL_INPUT` and `FCFL_OUTPUT`. Descriptors, U-Net masks and
federated runs are cached in the output directory, so an interrupted run continues where it stopped; a cache
built by different code is rebuilt. Setting `FCFL_SMOKE_TEST=1` runs every experiment in a few minutes with
reduced settings to check the installation; it does not reproduce the results.

| Experiment | Hardware | Approximate time |
|---|---|---|
| 1 | CPU or GPU | 1.5 to 2 h (descriptor computation about 1 h) |
| 2 | GPU | 1 to 1.5 h |
| 3 | CPU or GPU | 45 min |
| 4 | GPU | 1.5 h |
| 5 | GPU | 2.5 h |
| 6 | CPU or GPU | 10 min |

## Results

Figures from the runs reported in the thesis. All accuracies are balanced accuracies at patient level unless
stated otherwise; descriptor comparisons use the corrected repeated cross-validation t-test (Nadeau and
Bengio, 2003) and intervals over patients come from the bootstrap.

| Experiment | Result |
|---|---|
| 1 | 19 descriptors: 0.923 ± 0.006. Nested selection: 0.939 ± 0.038 per outer fold. Strictly tumour-only descriptors: 0.736 ± 0.013. Fractal-chaos family alone: 0.756 ± 0.014. Contribution of the fractal-chaos family: +13.2 points, 95% CI [7.2, 19.1], p < 0.0001 |
| 2 | Radiologist masks 0.979 [0.958, 0.996]; U-Net masks, intention-to-treat, 0.946 [0.915, 0.973]; head geometry alone 0.797 [0.746, 0.847] |
| 3 | 19 descriptors 0.860 ± 0.043; brain geometry 0.464 ± 0.044 |
| 4 | Damage at alpha 0.3: CNN 15.2 points, descriptors 1.3 points; gap +13.9 points, 95% CI [10.8, 17.1] over test patients. Federated learning on the descriptors: 0.856 slice level, 0.923 patient level |
| 5 | FedAvg 0.584 ± 0.074; FedRS 0.592 ± 0.072; FedLC 0.653 ± 0.026; class-presence restriction 0.649 ± 0.028; centralised 0.705 ± 0.011 (slice level, 20 seeds) |
| 6 | Full model: 0.897 with radiologist masks, 0.892 with U-Net masks; cost -0.6 points, 95% CI [-4.6, +3.7] |

Descriptor results are deterministic. U-Net training uses deterministic cuDNN settings in this code; the figure
of experiment 6 in the thesis came from an earlier run without them, so a rerun can differ from it slightly.

## Limits of the results

Glioma versus non-glioma approximates, but is not equivalent to, malignant versus benign. Descriptors are
computed from a tumour mask, and the federated comparison with the CNN uses radiologist masks. On Figshare,
which is not skull-stripped, the "brain" mask covers the head, and descriptors of the head region partly follow
the slice-selection shortcut described by Wallis and Buvat (2022); the strictly tumour-only figure is the
estimate robust to it. On BraTS, low-grade cases come mostly from one collection, so grade is partly confounded
with acquisition site.
