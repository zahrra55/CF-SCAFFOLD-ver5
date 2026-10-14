# Withdrawn claims and null results

Four lines of work produced results that were reported as null, or were
withdrawn after further testing. They are recorded here in full because they
shaped the final method, and because a reader checking the thesis against this
repository should find them rather than find a gap.

Each entry states what was claimed, what testing showed, and what was done about
it.

---

## 1. Clustered federated learning: withdrawn

**The claim.** Clients grouped by the distribution of the `C` descriptor, with a
separate model per cluster, beat a single global model by 12.5 points.

**Why it looked convincing.** The margin was large, the mechanism was plausible
(clients with similar tumor morphology should benefit from a shared model), and
it would have connected Part I to Part II, which was methodologically attractive.

**How it failed.** The original figure came from a single seed. Testing it
properly meant:

1. Repeating over 10 seeds with capacity-matched baselines, since three cluster
   models have three times the parameters of one global model and must be
   compared against an ensemble of the same size. The margin fell to a median of
   11.3 points but Holm-corrected `p` was 0.08 to 0.10.
2. Measuring the stability of the cluster assignment itself. Mean pairwise
   adjusted Rand index across seeds was **0.017**, meaning clients landed in
   essentially unrelated clusters from run to run.
3. Fixing the instability (anchored centroids, merging clusters below 10% of
   clients) and repeating over 20 seeds. The advantage **vanished**: Δ = +0.012,
   `p = 0.88`, winning on 10 of 20 seeds.

**Diagnosis.** The clustered model had degenerated to a near-constant predictor
at about 0.505. Its apparent wins occurred only on seeds where the global
baseline happened to collapse to chance. The margin was variance reduction on
unstable seeds, not a structural advantage.

**Action.** Withdrawn. It appears in the thesis only as a negative result, with
the ARI value quoted, because the low membership stability was the signal that
prompted the re-test and is worth recording as a diagnostic practice.

---

## 2. Per-patient personalisation: invalid by construction

**The claim.** Adapting the global model to each held-out patient, using the
Motley support/query protocol, improved balanced accuracy by 32 points with
Cohen's `d = 10.38`, and harmed 0% of patients.

**Why it was checked rather than reported.** Those numbers are not plausible.
Personalisation gains in the federated literature are typically a few points and
harm some fraction of clients. An effect size above 10 and a harm rate of exactly
zero indicate a broken evaluation, not a strong method.

**How it failed.** Each patient has **one tumor type across all their slices**.
Splitting a patient's slices into a support set and a query set therefore leaks
the label: the support labels determine the query labels.

This was confirmed with a decisive test. A predictor that **ignores the images
entirely** and copies the majority label of the support set scores balanced
accuracy **1.000** on the query set. No image information is required to solve
the task as posed.

**Action.** The result was discarded, not reported. The leakage test was kept as
a permanent check in the pipeline, and the episode is described in the manuscript
as evidence that the pathology lies in the output layer rather than in
representation. The broader point, that a standard personalisation protocol
becomes a leakage channel when each client holds a single label, is a
methodological observation worth stating.

**Code.** The check is `code/shared/leakage_guard.py`.

---

## 3. Bio-anchored contrastive learning: null

**The idea.** Add an auxiliary contrastive term to the local objective whose
targets come from the continuous descriptors of Part I (`C`, lacunarity, border
gradient) rather than from discrete labels. Because the term never references
labels, it would continue to organise the latent space for classes a client does
not hold, and it would connect the two parts of the thesis.

**How it was set up.** A success criterion was fixed **before the experiment ran**:
a gain of at least 2 points, `p < 0.05`, and wins on at least 13 of 20 seeds.
This was written into the script so that a marginal outcome could not be
reinterpreted afterwards.

Two errors in the first formulation were found and corrected before running:
multiplying the biophysical similarity by a label mask reintroduced the
missing-class problem the term was meant to avoid, and computing the kernel
bandwidth globally would have leaked statistics between clients. The final
version uses similarity alone, with the bandwidth taken as the median pairwise
distance within the local batch and descriptors normalised per client.

**Result (16 seeds, run stopped at the Kaggle session limit).**

| Arm | Balanced accuracy |
|---|---|
| Masking only | 0.640 ± 0.040 |
| Masking + contrastive (λ = 0.1) | 0.646 ± 0.040 |
| Masking + contrastive (λ = 0.5) | 0.652 ± 0.042 |
| Contrastive only, no masking | 0.539 ± 0.111 |

The best arm gained **1.26 points** (`p = 0.017`, 11 of 16 seeds). That is below
the preregistered threshold of 2 points, so the criterion was not met.

**Action.** Not claimed as a contribution. A later suggestion to add a non-linear
projection head and re-run in order to cross the threshold was declined, because
changing the design after seeing the result and then testing against the same
threshold is not a valid procedure.

**What was kept.** The `contrastive only` arm is the informative one. Without
masking, the contrastive term performed **10 points worse** than masking, with
two collapses and three times the variance. Organising the latent space does not
address the pathology; removing the absent-class gradient does. This appears in
the manuscript's limitations section as supporting evidence for the diagnosis.

**Code.** `code/pillar2_federated/stage7_bio_contrastive.py`

---

## 4. Aggregation weighting by C: null

**The claim tested.** Weighting client contributions during aggregation by the
divergence of their `C` distribution from the global pool would counteract drift.

**Result.** No improvement over SCAFFOLD (Δ = +0.012, Wilcoxon `p = 0.812`).

**Why the null is expected rather than surprising.** SCAFFOLD's control variates
already correct the drift that such a weight would target, so the weight is
largely redundant. Reported as a theory-consistent null.

---

## Later nulls and bounded results

Recorded in [`08_later_experiments.md`](08_later_experiments.md): the frozen
pretrained backbone shrinks rather than amplifies the gain (Stage 9/9B); the
ratio C adds nothing over Ft and Fb (Stage 10B); automatic active-contour
segmentation fails (Stage 11); real SCAFFOLD control variates change nothing
measurable (Stage 13). A proposed random-mask control was rejected before
running because, with three classes, it cannot avoid masking present labels.
In the final pipeline, active-contour refinement of the U-Net tumour region
lowered Dice on real images (-0.047, p = 2 × 10⁻¹²¹) and was dropped, although it
had helped on synthetic data. The claim that Part I exceeds Cheng et al. (2015)
was withdrawn after the feature corrections (0.897 against 0.913).

---

## Results of the fractal-chaos stages that did not support a claim

A first matched-damage test (Stage 17B) did not meet its rule and is kept in the record next to the corrected Stage 18 test. The class-presence correction lowered the accuracy of federated learning on descriptors (Stage 17). The fractal-chaos family is not the most robust descriptor group (Stage 18B). The tumour-boundary fractal dimension, the roughness exponent and the Higuchi dimension of the boundary signature were rejected (Stage 19). On BraTS, the fractal measures of the enhancing region and of necrosis added nothing beyond the presence of these regions (Stage 19), and the preregistered grading threshold for the fractal-chaos family was not met (Stage 17).

---

## Results not confirmed by the corrected statistics (Stage 20)

Under the corrected repeated cross-validation t-test, grayscale lacunarity (+2.38 points, p = 0.098) and FuzEn2D (non-inferiority not established) were not accepted, reversing two decisions of Stage 19. The margin of glioma-versus-non-glioma classification over the shortcut control (lower bound 0.091) missed the required 0.10, and the Stage 19 conclusion that BraTS presence flags carry more than the sub-region fractal measures was withdrawn (corrected p = 0.212). All are listed with their statistics in [`10_negative_results.md`](10_negative_results.md).

---

## What these four have in common

In each case an initial result looked strong and did not survive scrutiny. The
pattern was consistent enough to become a working rule: a large margin measured
on few seeds, an implausibly large effect size, or a metric that cannot fail are
all reasons to test harder rather than to write faster.

Three specific practices came out of this and were applied to everything
afterwards:

- **Multi-seed by default.** Twenty paired seeds, not one.
- **Preregistered thresholds.** For the contrastive experiment the criterion was
  fixed in the script before the run.
- **Adversarial checks.** For any evaluation, ask what a model that ignores the
  input would score. If the answer is high, the protocol is broken.
