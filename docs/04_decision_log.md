# Decision log

Every methodological choice, the reason for it, and the alternative that was not
taken. Arranged by the question an examiner is likely to ask.

---

## Data

**Why the Figshare dataset and not a larger one?**
It is the only public three-class brain tumor dataset that carries both expert
tumor masks and patient identifiers. The masks are required to define the
descriptor at all, since it is a ratio between two regions. The identifiers are
required for leakage-free evaluation. Larger datasets exist but lack one or both.

**Why BRISC 2025 for external validation and not the LGG segmentation dataset?**
LGG contains a single tumor type, so the label-skew condition under study cannot
occur in it. BRISC has the same three classes, which makes the federated setting
transferable. Its drawback, the absence of patient identifiers, is documented
and the results are reported as method-comparison only.

**Why 128 × 128?**
Large enough to preserve the structure the box-counting estimate measures, small
enough that 20 seeds × 4 methods × 40 rounds fits in the available compute. The
box-counting sizes used are powers of two up to 128, so the resolution and the
estimator are matched.

---

## Part I

**Why a ratio `Ft/Fb` rather than the tumor dimension `Ft`?**
A raw fractal dimension varies with scanner, sequence and noise, which differ
between patients. The tumor-free brain in the same slice shares those conditions,
so dividing removes the common factor. The ratio measures irregularity relative
to that patient's own brain, which is what makes it comparable across patients.

**Why the expert mask rather than automatic segmentation?**
To isolate the descriptor from segmentation error. Using an automatic mask would
mean any result confounds the quality of the descriptor with the quality of the
segmenter. The cost is that the method as evaluated needs a mask at inference,
which is stated as a limitation.

**Why a random forest and not a neural network?**
With 218 usable patients and 16 features, a forest is matched to the data. It
also permits feature-importance inspection, which is the point of using
interpretable features. Part II provides indirect support: a residual network
performed worse than a three-block CNN on the same data.

**Why 300 trees?**
Enough for the out-of-bag estimate to stabilise on a dataset of this size;
beyond that, added trees change results negligibly and cost time. Depth and leaf
size were tuned by grid search inside the training folds.

**Why `GroupKFold` and not stratified k-fold?**
Slices from one patient are near-duplicates. Stratified k-fold on slices would
place near-duplicates on both sides of the split. `GroupKFold` with the patient
identifier as the group prevents this.

**Why majority vote to patient level?**
The clinical unit of decision is the patient, not the slice. A patient-level
figure is also the only one comparable to Cheng et al. (2015).

---

## Metrics

**Why balanced accuracy rather than accuracy?**
The classes are uneven (708 / 1426 / 930). A classifier that always predicts
glioma gets 46.5% plain accuracy while being useless; the same classifier gets
0.333 balanced accuracy. Since Part II studies collapse onto a single class,
plain accuracy would conceal the phenomenon under study. Balanced accuracy is the
unweighted mean of per-class recall, so each class contributes one third
regardless of size.

**Balanced accuracy weights all classes equally. Is that always right?**
No, and this is stated as a limitation. It assumes symmetric error costs, which
does not hold in clinical screening where missing an aggressive tumor is worse
than a false alarm on a benign one. A cost-sensitive metric would need a
clinically validated cost matrix, which was not available. The scope of this work
is measuring discriminative capacity, not making treatment decisions.

**Why not AUC as the headline metric?**
AUC is reported per class for the descriptor analysis, where the question is how
well a single continuous feature separates one class from the rest. For the
multi-class model the decision is a hard assignment, so a metric on the decision
is the appropriate summary.

**Why is collapse defined at 0.40?**
Chance is 0.333 for three balanced classes. The threshold sits above chance but
below any run that is doing real work; the lowest non-collapsed value observed in
the corrected configurations is 0.598, so the threshold does not sit near any
observed value and the count is not sensitive to its exact position.

---

## Part II

**Why SCAFFOLD rather than FedAvg as the base optimiser?**
Its control variates correct drift in the shared feature extractor. That drift
is a real effect and is separate from the output-layer problem this work
addresses, so correcting it removes a confound.

**Why GroupNorm rather than BatchNorm?**
BatchNorm estimates statistics from local batches; under non-identical client
distributions those estimates diverge and averaging is unreliable. Hsieh et al.
(2020) showed group normalisation recovers most of the loss, and later
normalisation benchmarks reproduced it.

**Why Dirichlet(0.3) as the main setting?**
It produces the condition under study — 13 of 17 clients missing at least one
class — while leaving every client enough samples to train on. The sweep in
Stage 5 covers 0.1 and 3.0 as well, so the choice is a reported point on a
studied range rather than a single tuned value.

**Why 18 clients?**
Enough that the Dirichlet draw produces varied missing-class patterns; small
enough that each client retains a usable number of patients from 233.

**Why `α = 0` rather than a tuned `α`?**
Stage 5 measured `α ∈ {0, 0.5, 1}` across three skew levels and `α = 0` was best
at all of them. It also requires no tuning, which matters in a federated
deployment where no validation set spans all sites.

**Is `α = 0` novel?**
No, and the thesis says so. It is the limiting case of the FedRS scaling rule.
The contribution is identifying which property of these objectives produces the
improvement, and the regime in which the limiting case is the right choice.

**Why does the comparison with logit calibration matter if it is a tie?**
Because the tie is the finding. Two operators with different forms produce the
same outcome while partial scaling produces almost nothing, which places the
effect in the presence of the absent-class term rather than in any specific
remedy. Omitting the comparison would also have been the most likely single
cause of rejection.

**Why keep gradient clipping when it hurts on its own?**
It stabilises in combination with the restricted loss. Once class masking
conditions the loss landscape, clipping bounds residual gradient noise instead of
truncating a diverging signal. The ablation showing it hurting alone is reported
rather than hidden.

**Why 40 rounds?**
Longer schedules were tested in Stage 1 and increased the collapse rate on the
uncorrected objective. Forty rounds is where the corrected configurations have
converged without exposing the uncorrected ones to additional harm, which keeps
the comparison fair to the baselines.

**Why 20 seeds?**
The seed controls the partition, the split, the initialisation and the batch
order together, so runs are paired and differences can be tested with a paired
test. Twenty gives adequate power for the effect sizes observed while remaining
affordable.

---

## Statistics

**Why Wilcoxon signed-rank and not a t-test?**
The per-seed differences are not guaranteed normal, and with collapse runs the
distribution is bimodal. Wilcoxon is paired and makes no normality assumption.

**Why Holm correction?**
Each method is compared against three baselines, so three tests are run per
family. Holm controls the family-wise error rate and is uniformly more powerful
than Bonferroni.

**Why report Cohen's `d` and bootstrap intervals alongside `p`?**
A `p`-value indicates whether an effect is distinguishable from zero, not whether
it is large enough to matter. Effect size and interval answer the second
question. In this project two results were significant but small and were
reported as such rather than promoted.

**Why preregister a threshold for the contrastive experiment?**
Because three earlier results in this project had looked promising and did not
survive. Fixing the criterion in the script before the run removed the option of
reinterpreting a marginal outcome afterwards.
