# Part I — Interpretable classification from tumor morphology

This part follows the direction proposed by the supervisor: separate the tumor
from the surrounding brain, describe it with quantities that carry physical
meaning, and classify tumor type from those descriptions rather than from a
learned representation.

The record below is arranged in the order the work was actually carried out, so
that each stage can be read as an answer to the question raised by the one
before it.

---

## Stage 1 — Region separation

**Question.** A fractal dimension is a property of a region. To compare tumor
tissue against healthy tissue, the two regions must first be separated
unambiguously.

**Method.** Three regions are derived from each slice.

1. *Brain region.* The slice is thresholded at `max(0.25 × mean intensity, 10)`,
   holes are filled, a binary opening with two iterations removes speckle, and
   the largest connected component is retained. This discards the black
   background, which would otherwise dominate any box-counting measure with
   empty boxes.
2. *Tumor region.* The expert mask from the dataset, intersected with the brain
   region so that no mask pixel outside the estimated brain contributes.
3. *Tumor-free brain.* The brain region with the tumor region removed.

**Why the expert mask rather than automatic segmentation.** The descriptor is the
object under study; if it were computed on an automatically produced mask, any
result would confound the quality of the descriptor with the quality of the
segmentation. Using the provided expert masks isolates the descriptor. The
consequence, which must be acknowledged rather than hidden, is that the method
as evaluated requires a mask at inference time. A deployment path would need an
automatic region extraction step, and the accuracy of that step would need to be
reported separately.

**Rejection criteria.** A slice is discarded if the tumor region contains fewer
than 16 pixels or the tumor-free region fewer than 64, since a box-counting
estimate over a region that small is not meaningful. On the Figshare dataset
this leaves 2740 of 3064 slices when descriptors are required. Experiments that
do not need descriptors use all 3064.

**Code.** `code/shared/data_loading.py`, function `brain_mask_of`.

---

## Stage 2 — The fractal descriptor C = Ft / Fb

**Question.** Tumor tissue is structurally more irregular than healthy tissue,
and the degree of irregularity differs between tumor types. Can that be reduced
to a single number that means the same thing across patients?

**Method.** The fractal dimension is estimated by differential box counting. For
a region, intensities are rescaled to [0, 255] within that region, the plane is
covered by boxes of side `s ∈ {2, 4, ..., 128}`, and for each box the number of
intensity levels spanned is counted. The dimension is the slope of
`log N(s)` against `log(1/s)`.

The descriptor is the ratio

```
C = Ft / Fb
```

where `Ft` is the dimension of the tumor region and `Fb` that of the tumor-free
brain in the same slice.

**Why a ratio rather than the tumor dimension alone.** A raw fractal dimension
depends on acquisition parameters, scanner, contrast and noise, which vary
between patients and sites. The tumor-free brain in the same slice was acquired
under exactly those conditions, so dividing by it removes the shared factor.
`C` measures how irregular the tumor is *relative to that patient's own brain*,
which is why it is comparable across patients in a way that `Ft` is not. It is
also the reason the descriptor survives a change of dataset.

**Measured range.** `C ∈ [0.428, 1.076]` on the usable Figshare slices, with the
pituitary class concentrated at the low end.

**Prior work and the limit of the novelty claim.** Box-counting fractal analysis
of brain tumors dates to Iftekharuddin et al. (2003). Computing dimensions over
both tumorous and non-tumorous tissue has been done for glioma grading
(Frontiers in Physiology, 2023, PMC10390093), and fractal measures combined with
a regularity ratio appear in Sánchez and Martín-Landrove (2022). What appears not to
have been used before is the single tumor-to-brain ratio `Ft / Fb` as a
descriptor for three-class tumor typing. The claim is restricted to that and is
phrased as "to our knowledge" in the thesis, because a literature search cannot
establish absence.

**Code.** `code/shared/descriptors.py`, function `boxcount_fd`.

---

## Stage 3 — Does C carry signal, and is it size in disguise?

**Question.** A descriptor can appear discriminative because it correlates with
something trivial. Tumor types differ in typical size, and fractal estimates are
sensitive to region area. The descriptor must be tested against that explanation.

**Findings.**

*Per-class discriminative strength.* One-vs-rest AUC for `C` alone:

| Class | AUC |
|---|---|
| Pituitary | 0.907 |
| Glioma | 0.735 |
| Meningioma | 0.613 |

`C` separates pituitary adenomas strongly. It separates meningiomas barely above
chance. This is stated in the thesis as a limitation rather than omitted:
meningiomas are morphologically heterogeneous, and a single irregularity ratio
does not capture them. The full feature set compensates, but the descriptor's
own strength is uneven.

*Size confound.* After removing the component of `C` predictable from tumor area,
45% of its class separation remains. The descriptor is therefore partly
size-influenced and partly independent of size. Both halves of that sentence are
reported.

**Why this stage exists.** Without it, the honest answer to "is this just tumor
size?" would be unknown. With it, the answer is "partly, and here is how much."

---

## Stage 4 — The feature set

**Question.** `C` alone is insufficient, particularly for meningioma. What should
accompany it, and how is redundancy avoided?

**The sixteen features.**

| Group | Features |
|---|---|
| Fractal | `Fb`, `C` |
| Texture (GLCM) | `GLCM_t`, `GLCM_b`, `correlation_t` |
| Entropy | `Dispersion_t` (Shannon entropy of tumor intensities) |
| Lacunarity | `lacun_t`, `lacun_b` |
| Shape | `solidity`, `eccentricity`, `extent`, `circularity` |
| Edge | `edge_density`, `mean_grad`, `border_grad` |
| Size | `size` (tumor pixel count) |

**Why these.** Each group answers a question the others cannot. Fractal measures
capture irregularity of intensity structure; lacunarity captures how gappy the
region is at a given scale, which is different from how irregular it is; shape
features capture the outline, which is what distinguishes a well-circumscribed
meningioma from an infiltrating glioma; edge features capture the sharpness of
the boundary, which reflects infiltration. Size is included explicitly rather
than left implicit, so that the model can use it and the analyst can see it.

**Why interpretable features rather than a deep network.** Two reasons. Each
feature corresponds to something a radiologist can name, so a prediction can be
explained by pointing at a measurement rather than at a saliency map. And with
233 patients, a small feature set with a random forest is better matched to the
data than a network with millions of parameters, which is confirmed later in
Part II where a residual network performs worse than a three-block CNN on the
same data.

**Code.** `code/pillar1_classification/extract_features.py`.

---

## Stage 5 — Classification and evaluation

**Question.** How well does the feature set classify tumor type, measured in a
way that does not fool itself?

**Protocol.**

- Random forest, 300 trees, with `max_depth` and `min_samples_leaf` selected by
  grid search *inside* the training folds only.
- Five-fold `GroupKFold` with the patient identifier as the grouping variable.
  All slices from a patient stay on the same side of every split.
- The inner grid search also uses `GroupKFold`, so hyperparameter selection
  cannot leak across patients either.
- Slice predictions are aggregated to a patient-level prediction by majority
  vote, and the reported figure is patient-level balanced accuracy.

**Why patient-level and not slice-level.** Slice-level numbers on this dataset
are not comparable to patient-level numbers, and the difference is large. This is
the single most important methodological point in Part I and is defended in the
thesis with the Yagis et al. (2021) measurement described in
[`00_datasets.md`](00_datasets.md).

**Why balanced accuracy and not accuracy.** The classes are uneven (708 / 1426 /
930). A classifier that always predicts glioma obtains 46.5% plain accuracy while
being diagnostically useless; the same classifier obtains 0.333 balanced
accuracy. Since Part II studies a failure mode that is exactly collapse onto one
class, plain accuracy would hide the phenomenon under study, and the same metric
is used in both parts for consistency.

**Result.** Approximately 0.92 patient-level balanced accuracy, against the
Cheng et al. (2015) reference of 0.913 on the same dataset.

**Honest positioning against the literature.** Two later published results on
this dataset are higher: Gumaei et al. (2019) report 94.23% with hybrid features
and a regularised extreme learning machine, and Deepak and Ameer (2019) report
98% with GoogLeNet features under patient-level five-fold cross-validation. The
result here is therefore competitive with the original benchmark but not the
highest reported. What it offers instead is sixteen named, measurable features
in place of a learned representation. That trade-off is the argument, and
overstating the accuracy would invite a reviewer to make the comparison for us.

**Code.** `code/pillar1_classification/classify.py`.

---

## Stages 10–12 — later tests that narrowed the claims

Three later experiments changed what Part I can claim. Over ten repeated
patient-grouped splits, the ratio C adds nothing on top of Ft and Fb
(−0.08 pp, p = 0.93) and is worse when used instead of them (−1.19 pp,
p = 0.004), while the fractal family as a whole adds +1.69 pp (p = 0.002). C is
therefore presented as an interpretable member of the fractal family, not as a
superior descriptor. Automatic delineation by active contour failed (median
Dice 0.000) because the seed does not find the tumour, which makes the expert
mask an established requirement. Leakage from slice-level splitting was
measured at +3.38 points on this feature set. Details and tables in
[`08_later_experiments.md`](08_later_experiments.md).

---

## Summary of Part I

| Item | Value |
|---|---|
| Usable slices | 2740 of 3064 (descriptors require a valid mask) |
| Patients | 218 of 233 |
| Features | 16 |
| Classifier | random forest, 300 trees |
| Evaluation | patient-grouped 5-fold, majority vote to patient level |
| Metric | balanced accuracy |
| Result | ~0.92 |
| Reference | Cheng et al. 2015, 0.913 |

**Known limitations, all stated in the thesis.**

1. Requires an expert tumor mask at inference time.
2. `C` alone discriminates meningioma weakly (AUC 0.613).
3. `C` is partly confounded with tumor size (45% of separation survives removal).
4. Single modality, single dataset, single institution of origin.
5. Two published methods report higher accuracy on the same data.
