# Later experiments (Stages 9–13)

These experiments were run after the manuscripts were first drafted. Each one
tested a specific claim, and several of them narrowed what the thesis can say.
They are recorded here in the order they were run.

Stage numbers follow the session record and the script filenames. They continue
after the numbering in `02_pillar2_federated.md`, which covers the earlier
federated stages.

---

## Stage 9 / 9B — Does the correction still matter with a frozen pretrained backbone?

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

## Stage 10 / 10B — Is C = Ft/Fb better than its two components?

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

## Stage 11 — Automatic segmentation by active contour

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

## Stage 12 — Audit of an alternative DBC implementation, and symmetry seeding

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

## Stage 13 — Real SCAFFOLD control variates

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
