# Datasets

Two public datasets are used. Neither is redistributed in this repository.

---

## 1. Figshare brain tumor dataset (primary)

The dataset published alongside Cheng et al. (2015), used for all primary
experiments in both parts of the thesis.

| Property | Value |
|---|---|
| Slices | 3064 |
| Patients | 233 |
| Classes | meningioma (708), glioma (1426), pituitary (930) |
| Modality | T1-weighted, contrast-enhanced |
| Annotations | expert tumor masks, tumor border coordinates |
| Patient identifiers | present (`PID` field) |
| Format | MATLAB v7.3 (HDF5) `.mat`, one file per slice |
| Licence | CC BY 4.0 |

**Source:** https://doi.org/10.6084/m9.figshare.1512427, **version 5**, posted
2 April 2017, is the release used here and the one cited throughout the
literature. Later versions (6 to 8) were posted on 21 December 2024.

**Reference publication:** J. Cheng, W. Huang, S. Cao, R. Yang, W. Yang, Z. Yun,
Z. Wang, and Q. Feng, "Enhanced performance of brain tumor classification via
tumor region augmentation and partition," *PLoS ONE* 10(10), art. e0140381,
2015, doi: 10.1371/journal.pone.0140381. The dataset page also cites the
companion retrieval paper, *PLoS ONE* 11(6), art. e0157112, 2016.

**A DOI to avoid.** `10.1371/journal.pone.0139642` is sometimes given for this
dataset. It is not a Cheng paper: it is "Assessing the Power of Exome Chips", an
unrelated genetics study in the same PLOS ONE issue. The adjacent article
numbers are the likely source of the confusion.
**Mirror used for compute:** `ashkhagan/figshare-brain-tumor-dataset` on Kaggle.

### Why this dataset

Three properties made it the only viable choice for this work.

It carries **expert tumor masks**. Part I computes a fractal dimension over the
tumor region and separately over the brain with the tumor removed. Without a mask
the two regions cannot be separated, and the descriptor cannot be defined.

It carries **patient identifiers**. Adjacent slices from one patient are visually
near-identical, so a random slice-level split places correlated images in both
training and test sets. Yagis et al. (2021) measured this effect on brain MRI and
found that slice-level splitting inflated accuracy by 29 to 55 points relative to
subject-level splitting, and that randomly labelled scans reached about 96%
accuracy under a slice-level split against 50% under a subject-level split.
That study measured Alzheimer's and Parkinson's cohorts rather than tumor data;
the mechanism depends on one subject contributing many adjacent slices, which
holds here too, and the citation should be made on those terms. The
`PID` field is what makes patient-grouped evaluation possible here.

It is the **standard benchmark** for three-class tumor-type classification, so
the classification result is directly comparable to published figures.

### File structure

Each `.mat` file contains one struct, `cjdata`, with fields:

| Field | Contents |
|---|---|
| `label` | 1 = meningioma, 2 = glioma, 3 = pituitary |
| `image` | grayscale slice |
| `tumorMask` | binary mask of the tumor region |
| `tumorBorder` | border coordinates |
| `PID` | patient identifier, stored as character codes |

Two practical notes that cost time to discover and are recorded here so they do
not have to be rediscovered:

- The files are MATLAB v7.3, which is HDF5. They must be read with `h5py`, not
  `scipy.io.loadmat`. Arrays come out **transposed** relative to the MATLAB
  orientation and must be transposed back.
- `PID` is stored as an array of character codes, not a string. It is recovered
  with `"".join(chr(int(c)) for c in raw if int(c) > 0)`.

Both behaviours are implemented in `code/shared/data_loading.py`.

---

## 2. BRISC 2025 (external validation)

Used only to test whether the federated finding holds outside the primary
dataset.

| Property | Value |
|---|---|
| Slices used | 3000 (1000 per class, drawn from the segmentation subset) |
| Classes | glioma, meningioma, pituitary (the `no_tumor` class is excluded) |
| Annotations | expert segmentation masks |
| Patient identifiers | **not distributed** |
| Format | `.jpg` images with matching `.png` masks |

**Source:** `briscdataset/brisc2025` on Kaggle.
**Reference publication:** A. Fateh, Y. Rezvani, S. Moayedi, S. Rezvani,
F. Fateh, M. Fateh, and V. Abolghasemi, "BRISC: Annotated Dataset for Brain
Tumor Segmentation and Classification," *Scientific Data*, vol. 13, art. 361,
5 February 2026, doi: 10.1038/s41597-026-06753-y. Companion preprint:
arXiv:2506.14318.

Note that "volume 12, article 845 (2025)" appears in some secondary listings for
this dataset. That is a different paper, which appears inside BRISC's own
reference list, and it should not be used.

### Structure

```
segmentation_task/{train,test}/images/*.jpg
segmentation_task/{train,test}/masks/*.png     (same basename as the image)
classification_task/{train,test}/{glioma,meningioma,pituitary,no_tumor}/*.jpg
```

Filenames follow `brisc2025_<split>_<index>_<code>_<plane>_<sequence>.<ext>`,
where the tumor code is `gl`, `me`, `pi` or `no`. In the segmentation subset the
class is therefore encoded in the **filename**, not in the parent directory, and
the loader extracts it from there.

### Two limitations that must be stated when this dataset is cited

**No patient identifiers.** Clients cannot be formed by patient, so on this
dataset they are formed by drawing slices. Slices from one patient may fall on
both sides of the train/test split. The BRISC numbers are therefore valid for
comparing methods against each other, since every method faces the same
partition, but they are not comparable in absolute terms to the patient-level
Figshare numbers.

**BRISC is not independent of Figshare.** The dataset publication states that
the images were collated from existing public datasets, and the dataset
documentation lists the SARTAJ and Br35H collections among its sources. These
are the same collections that, together with the Cheng data, make up the
Nickparvar aggregate. BRISC therefore shares source material with the primary
dataset used here.

What remains valid is a robustness check rather than external validation: the
masks are newly annotated by the BRISC authors, the class balance differs, and
the partitioning differs. Any claim that BRISC constitutes an independent test
set must be removed from the manuscript and the thesis. This was verified
against the dataset publication rather than assumed.

---

## Preprocessing applied to both datasets

Identical for all experiments unless a script states otherwise.

1. Grayscale, resized to 128 × 128 (bilinear for images, nearest for masks so
   that mask values stay binary).
2. Intensity scaled to [0, 1] by min-max over the individual slice.
3. Masks thresholded at 127 after resizing.
4. For Part I only: the brain region is estimated by thresholding, hole filling
   and keeping the largest connected component; the tumor region is intersected
   with it, and the complement within the brain gives the tumor-free region.

Implemented in `code/shared/data_loading.py` and `code/shared/descriptors.py`.

---

## Access

Both datasets require a free Kaggle account. In a Kaggle notebook, attach them
through **Add Input**, selecting the entry marked *Dataset* rather than one
marked *Notebook* or *Code*. The loaders in this repository search
`/kaggle/input` recursively with `os.walk`, so they locate the files regardless
of the directory depth at which Kaggle mounts them.
