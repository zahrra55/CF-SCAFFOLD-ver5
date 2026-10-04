# Later experiments (Stages 9–13)

These experiments were run after the manuscripts were first drafted. Each one
tested a specific claim, and several of them narrowed what the thesis can say.
They are recorded here in the order they were run.

Stage numbers follow the session record and the script filenames. They continue
after the numbering in `02_pillar2_federated.md`, which covers the earlier
federated stages.

---

## Stage 9 / 9B: Does the correction still matter with a frozen pretrained backbone?

**Question.** If an ImageNet-pretrained ResNet-18 is frozen and only the
classifier is trained federatedly, the shared feature extractor cannot drift.
Does the benefit of absent-class masking grow, shrink or disappear?

**Prediction before running.** The gain would grow, because the output layer
would then carry all of the damage.

**Result.** The prediction failed. The FedAvg baseline itself rises sharply
(Figshare 0.745, BRISC 0.804), and the gain over FedAvg shrinks to about
+2.2 points on Figshare and +1.9 on BRISC. The corrections (masking, FedRS,
logit calibration) become statistically indistinguishable from each other, with
logit calibration about 0.3–0.4 points ahead.

**Reading.** A strong fixed representation removes most of the heterogeneity
damage before the output layer is reached. This agrees with Nguyen et al.
(ICLR 2023) and Chen et al. (ICLR 2023) on pre-training in federated learning.
The frozen-backbone observation is anticipated by Park et al. (OvA-LP, 2025) and
is consistent with Haseeb et al. (2025); it is reported as a bounded finding,
not as a contribution.

**Code.** `code/pillar2_federated/stage9_frozen_backbone.py`,
`stage9b_frozen_backbone_brisc.py`

---

## Stage 10 / 10B: Is C = Ft/Fb better than its two components?

Stage 10 used one deterministic split and suggested that adding C helped by
about 1.3 points. Stage 10B repeated every comparison over ten patient-grouped
splits.

| Feature set | Features | Patient-level balanced accuracy (10 splits) |
|---|---|---|
| Ft, Fb | 2 | 0.6813 ± 0.0110 |
| C alone | 1 | 0.5892 ± 0.0139 |
| 14 others, no fractal | 14 | 0.9045 ± 0.0108 |
| 14 others + Ft, Fb | 16 | **0.9222 ± 0.0068** |
| 14 others + C | 15 | 0.9103 ± 0.0072 |
| 14 others + Ft, Fb, C | 17 | 0.9213 ± 0.0092 |

Paired over the same splits:

| Question | Δ | Wilcoxon p | Wins |
|---|---|---|---|
| Does C add anything on top of Ft, Fb? | −0.08 pp | 0.93 | 3/10 |
| Can C replace Ft, Fb? | −1.19 pp | 0.004 | 0/10 |
| Does the fractal family help at all? | +1.69 pp | 0.002 | 10/10 |

**What this establishes.** The fractal dimensions contribute a real,
reproducible gain. The ratio does not: it adds nothing to the pair and is worse
than the pair when used instead of it, because dividing discards the absolute
scale of both dimensions. The earlier +1.3 point figure was split-to-split noise
(the spread across splits is ±0.7 to ±1.1 points). Battalapalli et al. (2023),
who used the two dimensions as separate features, made the better choice.

What remains for C is interpretability and one robustness property measured
earlier: it is about 30 % less variable than Ft under simulated acquisition
perturbation (p = 1.3 × 10⁻¹⁰). That perturbation was synthetic, not
multi-scanner.

**Consequence for the thesis.** C is presented as an interpretable member of the
fractal feature family, alongside Ft and Fb, not as a descriptor superior to
them.

Data: `results/tables/descriptor_ablation_stage10b.csv`

### Leakage, measured on this dataset

The same stage measured how much a slice-level random split inflates accuracy
here, with both arms scored at slice level so that only the split differs.

| Split | Balanced accuracy (10 splits) |
|---|---|
| Patient-grouped | 0.8571 ± 0.0073 |
| Random slice split | 0.8908 ± 0.0024 |
| Inflation | **+3.38 ± 0.79 points**, Wilcoxon p = 0.002, 10/10 |

Yagis et al. (2021) measured 29–55 points with 2D CNNs on neurodegenerative MRI.
Sixteen hand-crafted features cannot memorise an individual slice the way a
network with millions of parameters can, so the inflation is roughly nine times
smaller. The size of the leakage depends on model capacity, not only on the
split.

A separate effect was also quantified: majority voting from slices to patient
adds +6.43 points by itself. It is a property of the protocol, not of the
features, and slice-level and patient-level numbers must never be compared
directly.

Data: `results/tables/leakage_measurement_stage10b.csv`
Code: `code/pillar1_classification/stage10_descriptor_tests.py`,
`stage10b_repeated_splits_and_leakage.py`

---

## Stage 11: Automatic segmentation by active contour

**Question (proposed by the supervisor).** Can the expert tumour mask be replaced
by an automatic delineation, so the method runs without a radiologist?

**Method.** Morphological Chan–Vese active contour (Márquez-Neila et al., 2014,
as implemented in scikit-image), seeded from the brightest region inside the
estimated brain.

**Result.** Failure. Mean Dice 0.034, median 0.000; only 3.3 % of slices exceed
Dice 0.7. On most slices the automatic contour does not touch the tumour.

The initialisation test isolates the cause. With the same evolution rule:

| Initialisation | Dice (300 slices) |
|---|---|
| Checkerboard | fails on 299/300 |
| Centre disk | 0.057 |
| Brightest-region seed | 0.049 |
| Expert centroid (not usable in practice) | **0.656** |

The contour works once it starts inside the tumour. Locating the tumour is the
problem, not delineating it: on contrast-enhanced T1, vessels, choroid plexus
and normally enhancing tissue compete with the lesion for "brightest region".

**A result that looked wrong and was explained.** The pair (Ft, Fb) classified
almost as well from the failed automatic masks (0.6706) as from expert masks
(0.6680). Correlations explain it: Ft from automatic vs expert masks correlates
at −0.031 (no relation), Fb at +0.974 (unchanged), because removing a small wrong
region from the brain barely changes a mask covering most of the brain. The
class signal of the pair is carried largely by Fb.

**Consequence.** Dependence on an expert mask is an established limitation of
Part I, not a theoretical caveat. A learned localiser (for example nnU-Net) is
the route to full automation and is left to future work.

Data: `results/tables/active_contour_stage11.csv`
Code: `code/pillar1_classification/stage11_active_contour.py`

---

## Stage 12: Audit of an alternative DBC implementation, and symmetry seeding

**Part A: audit.** A colleague's differential box-counting script reported an
extremely clean separation of the three classes by C (ANOVA p = 4 × 10⁻²⁷²). Two
defects were found: raw MRI intensity cast directly to `uint8` for the brain
mask (values above 255 wrap around), and normalisation over a zero-filled array
instead of the region's own range. A second version of that script computed all
features at the native 512 × 512 resolution while the rejection thresholds were
calibrated for 128 × 128, so almost no slice was rejected (3064/3065 kept) and
meningioma AUC fell below chance (0.467). Resizing before region separation
restored the validated filtering rate exactly (2740 slices, 89.4 %) and
meningioma AUC to 0.621. The corrected script is kept as an audit record, not as
a replacement for the canonical pipeline.

Code: `code/audits/colleague_pipeline_resolution_fix.py`

**Part B: bilateral-symmetry seeding.** An alternative seed for the contour: the
connected region of highest left–right asymmetry, which does not assume the
tumour is bright. On synthetic data it reached Dice 0.658 on a bright lesion and
0.622 on a dark one, where brightest-region seeding scored 0.000. Two
implementation fixes were needed before it worked (an Otsu threshold instead of
a fixed percentile, and selecting the component with the highest mean asymmetry
instead of the largest one). **A run on the real dataset is not recorded in this
repository**, so no real-data claim is made for this seed.

Code: `code/pillar1_classification/stage12_dbc_audit_and_symmetry_seed.py`

---

## Stage 13: Real SCAFFOLD control variates

**Why it was run.** A technical review noted that the main ablation code,
although the project is named after SCAFFOLD, aggregates by weighted FedAvg and
has no control variates. The question was whether to fix the code or the name.

**Implementation.** SCAFFOLD Option II (Karimireddy et al., ICML 2020):
persistent per-client control variates `c_i` and a global `c`; the correction
`c − c_i` is added to every local gradient; after `K` local steps
`c_i ← c_i − c + (x − y_i)/(K·η)`; the global variate is updated scaled by the
participating fraction. Four silent-failure modes were guarded against and
checked: the sign of the correction, persistence of `c_i` across rounds under a
stable client id, scaling by participation, and clients dropped by the partition
never entering the state dictionary. The norm of `c` was traced across rounds to
confirm the state evolves rather than staying at zero.

**Preregistered criterion (fixed before running).** Adopt SCAFFOLD only if it
lowers mean accuracy by no more than one point and adds no collapse.

**Result, twice.**

| Test | FedAvg + restricted loss | SCAFFOLD + restricted loss | Δ | p | Cohen's d |
|---|---|---|---|---|---|
| Standalone, 20 seeds, 40 rounds | 0.6494 ± 0.0264 | 0.6520 ± 0.0296 | +0.25 pp | 0.546 | 0.21 |
| Integrated eighth ablation row, 20 seeds, 60 rounds | 0.657 ± 0.047 | 0.661 ± 0.051 | +0.43 pp | 0.230 | 0.12 |

No collapse in either arm. Both criteria passed, so real SCAFFOLD was added to the
canonical pipeline as an eighth, additive configuration. With `scaffold=False`
(the default) the code path is bit-identical to the one that produced every
earlier number; this was checked by running the old and new functions on the
same seed and data (identical output, 0.309524).

**Reading.** Control variates correct drift in the shared feature extractor; the
restricted loss removes the gradient on absent-class logits in the output layer.
That adding control variates changes nothing measurable supports the thesis's
account that, in this regime, the damage sits in the output layer.

Data: `results/tables/real_scaffold_stage13.csv`,
`results/tables/ablation_with_scaffold_summary.csv`
Code: `code/pillar2_federated/stage13_real_scaffold.py`; the eighth row runs from
`code/pillar1_classification/full_pipeline.py`.

---

## Rejected without running: a random-mask control

A proposed control replaced each client's true class mask with a random wrong
mask of the same size. With three classes this cannot be done cleanly: every
wrong mask of the same size necessarily masks out a class the client actually
holds, and masking a present label drives that sample's loss to about 10⁹. The
"random" arm would test training with impossible labels, not whether knowing the
true class set matters. The α sweep (Stage 5) answers the intended question
without masking any present class.

---

## The final pipeline run (corrected features)

Before the work was handed to the supervisor, the feature code was audited line
by line and four defects were corrected. The entropy feature used a histogram
density instead of probabilities, so its value followed the intensity range of
the region rather than the shape of its distribution. The texture features were
computed on the whole image with the background set to zero, so they mainly
measured tumour size and border. The differential box counting used a box height
of s instead of the s·G/M of Sarkar and Chaudhuri. Finally, Ft was missing from
the feature set although the descriptor is presented as part of the Ft, Fb, C
family. Each correction was verified on a test that the old code fails, and the
federated rows 1 to 7 were checked to be bit-identical to the documented ones.

With the corrected features the model reaches 0.897 patient-level and 0.827
slice-level balanced accuracy, which supersedes the earlier 0.919. The
one-vs-rest AUC of C (0.617, 0.729, 0.902 for meningioma, glioma and pituitary
tumours) barely moved, which indicates that the descriptor itself was not an
artefact of the defects. The federated ablation reproduced for the third time:
the class-presence correction adds 10.5 points (d = 1.58, 20 of 20 seeds), FedProx
adds nothing, and real SCAFFOLD with the corrected global control variate adds
0.38 points (p = 0.43).

The same run introduced automatic delineation. A small U-Net trained per fold to
locate the tumour reached a median Dice of 0.820, and replacing the radiologist
mask by the U-Net mask left patient-level accuracy essentially unchanged
(0.903 against 0.902). Refining the U-Net region with a morphological Chan-Vese
contour, which had helped on synthetic data, lowered Dice on the real images
(mean difference -0.047, p = 2 × 10⁻¹²¹) and was rejected.

Data: `results/tables/pillar1_final_pipeline.csv`,
`results/tables/segmentation_unet_final_pipeline.csv`,
`results/tables/ablation_with_scaffold_summary.csv`.

---

## Stage 14: is Fb measuring the tumour type or the slice?

Fb, the fractal dimension of the brain outside the tumour, is the strongest single
descriptor (pituitary AUC 0.933) and is almost unchanged by the mask (r = 0.997),
which raised the question of whether it describes the slice rather than the
tumour. Eleven geometric descriptors of the brain mask were computed, none of
which looks at the tumour.

Geometry alone predicts the tumour type with 0.750 patient-level balanced
accuracy, and brain-section area alone separates pituitary slices with an AUC of
0.879. Geometry explains 84% of the variance of Fb. This shortcut is not a new
observation: Wallis and Buvat (2022) showed on the same dataset that the tumour
type can be predicted from the skull outline (0.74) or from three non-tissue
features (0.77), and attributed it to the selection of representative slices.
Stage 14 is an independent replication with a different feature family and a
patient-level balanced metric.

Removing from each descriptor the part predicted by geometry collapsed the
pituitary AUC of Fb (0.933 to 0.595), but also of Ft (0.853 to 0.602), which is
computed inside the tumour. Because geometry itself carries class information,
this subtraction removes genuine signal as well, and the test could not separate
the two explanations. The decision rule printed "confound supported", but that
verdict overstates what the test can show, and Stage 14B was designed to answer
the question without subtraction.

Data: `results/tables/stage14_fb_geometry.csv`

---

## Stage 14B: the fractal descriptors inside bins of brain-section area

Slices were grouped into narrow bins of brain-section area and each descriptor
was compared only between classes inside the same bin, where section size cannot
explain a difference. With ten bins the control (brain area within its own bins)
remained at 0.608, so the preregistered ten-bin analysis did not fully remove
the size effect for pituitary slices; this is disclosed. With twenty bins the
control fell to 0.518 and the conclusion was unchanged.

| Pituitary vs rest | Raw AUC | Within 20 area bins |
|---|---|---|
| Fb | 0.933 | 0.824 |
| Ft | 0.853 | 0.876 |
| C | 0.902 | 0.886 |
| Brain area (control) | 0.879 | 0.518 |

The values of Fb follow brain-section area closely (r = 0.90), as expected from
the finite-size behaviour of box counting, yet its pituitary discrimination
survives size matching. For Ft and C the discrimination is essentially untouched,
so it is not a by-product of section size. Within the same bins C separates
glioma moderately (0.651) and meningioma barely (0.581).

Data: `results/tables/stage14b_area_matched.csv`

---

## Stage 15: head-to-head with Wallis and Buvat, and reliance on the shortcut

Two of the three features of Wallis and Buvat were reproduced with their
preprocessing (zero-pixel count and maximum intensity); the third, slice
orientation, is not stored in the Figshare files and was not guessed. Every model
was evaluated under one protocol (patient-grouped stratified five-fold, five
repeats).

| Model | Patient-level balanced accuracy |
|---|---|
| Wallis and Buvat, 2 features, linear SVM | 0.742 |
| Brain-mask geometry, 11 features | 0.750 |
| All shortcut variables together | 0.857 |
| Tumour-region features only (13) | 0.809 |
| Full model (17) | 0.917 |
| Full model plus shortcut variables | 0.938 |

On this dataset the information that has nothing to do with the tumour predicts
its type better than the tumour-region features do, and adding it to the full
model raises accuracy by 2.1 points, which illustrates how whole-slice models can
gain accuracy that is not about the tumour.

To ask whether each feature family relies on the shortcut, test slices were
placed in strata of near-identical shortcut information (k-means with k = 20 on
the four strongest shortcut variables) and balanced accuracy was measured inside
the strata. The tumour-region features kept 0.97 of their above-chance accuracy
(95% CI 0.87 to 1.06), against 0.65 for the brain-region features and 0.68 to
0.74 for the shortcut models themselves; stratifying by brain-area deciles gave
the same picture. The shortcut models did not fall to chance inside the strata,
so the strata did not hold the whole shortcut fixed, and the claim is therefore
comparative: tumour-region features depend markedly less on the shortcut than
any other feature family. The primary strata covered 1312 of 2740 slices; the
area-decile strata covered 2193.

A second literature search found no earlier publication of this comparison.
Shortcuts from file metadata on the merged Kaggle corpus were reported in a recent
preprint (Vangala et al., 2026).

Data: `results/tables/stage15_shortcut_reliance.csv`

---

## Stage 16: radiologist-free classification with tumour-region features

Stage 15 raised the question of whether the small cost of automatic masks came
from Fb, which barely changes with the mask and partly rides the shortcut. The
same U-Net masks were fed to two classifiers trained on radiologist-mask features,
and the cost of automation was estimated with a paired bootstrap over patients.

| Feature set | Radiologist masks | U-Net masks | Cost (95% CI) |
|---|---|---|---|
| Tumour-region only (13) | 0.813 | 0.775 | -3.8 points (-7.9 to 0.0) |
| Full model (17), control | 0.897 | 0.892 | -0.6 points (-4.6 to +3.7) |

The control reproduced the earlier result, so the comparison is valid. The
tumour-region features lose about four points, and the agreement of each feature
between radiologist and U-Net masks shows why: shape features agree poorly
(r = 0.25 to 0.45), texture and intensity features moderately (0.63 to 0.71), while
Ft and the gradient features hold (0.79 to 0.89). This pattern is consistent with
earlier studies of radiomic feature stability under different segmentations
(Tixier et al., 2019; Haarburger et al., 2020; Poirot et al., 2022; Nadeem et al.,
2024). The interval of the full model also shows that the earlier statement of a
0.2-point cost was more precise than the data allow: the correct reading is no
detectable cost, with an uncertainty of about four points.

Removing the shape features and re-running would probably give a set that is both
shortcut-robust and mask-robust, but choosing features after seeing these results
on the same data would not be a valid test, so it is left as a hypothesis for
future work.

Data: `results/tables/stage16_radiologist_free.csv`
