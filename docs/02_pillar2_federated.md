# Part II: Federated learning under missing classes

This part asks what happens when the classification task of Part I is
distributed across institutions that cannot share images, and why the obvious
approach fails.

The stages are numbered in the order they were run. Several of them exist only
because the stage before produced a result that could not be trusted.

---

## The setting

Training patients are distributed over 18 simulated clients by drawing, for each
class, a vector of proportions from a Dirichlet distribution with concentration
`α_dir`. Clients receiving fewer than four samples are dropped, which leaves 17
active clients at `α_dir = 0.3`.

At that setting **13 of 17 clients hold fewer than three tumor types**. Four hold
all three, nine are missing one, four are missing two. The mean fraction of
classes absent per client, written `ā` throughout, is 0.31.

Test patients are held out before partitioning, and all slices from a patient
stay together, so the patient-level protocol of Part I is preserved.

Two symbols are used and must not be confused:

- `α_dir`, the Dirichlet concentration controlling how skewed the partition is.
- `α`, the scaling factor applied to absent-class logits in the local loss.

---

## Stage 1: Which intervention actually stabilises training

**Question.** The federated baseline is unstable. Standard remedies exist. Which
of them work here?

**Method.** Six interventions were enabled one at a time on top of an
uncorrected baseline, 20 seeds each. A run is counted as collapsed if balanced
accuracy is at or below 0.40.

**Result.**

| Configuration | Balanced accuracy | Collapses |
|---|---|---|
| Baseline | 0.551 ± 0.031 | 0/20 |
| + gradient clipping | 0.529 ± 0.052 | 1/20 |
| + warmup and cosine decay | 0.371 ± 0.065 | **15/20** |
| + extended training | 0.379 ± 0.068 | **14/20** |
| + class-balanced sampling | 0.544 ± 0.081 | 1/20 |
| + restricted softmax | **0.611 ± 0.055** | 0/20 |
| + FedProx term | 0.611 ± 0.055 | 0/20 |

**What this established.** Only the restricted softmax helped. Two widely
recommended stabilisers made things dramatically worse, and FedProx added
nothing on top of the correction.

**Why the failures matter.** Warmup with cosine decay and extended training both
increase the number of local steps taken against an objective that still
contains the absent-class term. If that term is the problem, taking more steps
against it should make matters worse, and it does. The failures are therefore
evidence for the diagnosis, not noise, and they are reported in the manuscript
for that reason.

**Code.** `code/pillar2_federated/stage1_ablation.py`,
`code/pillar2_federated/stage1b_statistics.py`

---

## Stage 2: Why the loss term is the problem

**The argument.** For cross-entropy, the derivative of the loss with respect to
the logit of class `c` is

```
∂L/∂z_c = p_c − 1[c = y]
```

If `c` is absent from a client, `1[c = y] = 0` for every sample it holds, so the
derivative is `p_c > 0` at every step. The output parameters for that class are
driven in one direction throughout local training, with no opposing signal. The
size and direction of that displacement depend on *which* classes each client
lacks, which is fixed by the partition draw and therefore by the seed. Averaging
models displaced along different class directions produces a biased global
classifier, and the bias varies from run to run.

This explains both observations in Stage 1: the loss of mean accuracy, and the
large variance across seeds.

**The correction.** Scale absent-class logits by `α`:

```
L_k = −log( exp(z_y) / Σ_c s_c·exp(z_c) ),    s_c = 1 if c present, α if absent
```

The derivative on an absent-class logit becomes `α·p̃_c`, so `α` controls the
rate of displacement directly. At `α = 0` the term vanishes and the parameters
return to the server unchanged.

`α = 1` recovers ordinary cross-entropy. Intermediate `α` is the FedRS rule of
Li and Zhan (2021).

---

## Stage 3: Comparison against FedRS

**Question.** The masking objective is the `α → 0` limit of a published method.
Is the limit actually better than the published setting?

**Method.** Four objectives, identical partitions, 20 seeds, 40 rounds.

**Result (Figshare).**

| Method | Balanced accuracy | Worst seed |
|---|---|---|
| FedAvg | 0.585 ± 0.074 | 0.448 |
| FedRS (α = 0.5) | 0.591 ± 0.073 | 0.447 |
| FedRS (α = 0.25) | 0.605 ± 0.056 | 0.512 |
| Masking (α = 0) | **0.649 ± 0.026** | 0.604 |

Masking exceeded FedRS at its published setting by 5.8 points, Holm-corrected
`p = 0.002`, Cohen's `d = 0.86`, winning on 15 of 20 seeds. The across-seed
standard deviation fell by roughly a factor of three.

**Code.** `code/pillar2_federated/stage3_fedrs_comparison.py`

---

## Stage 4: External validation on BRISC 2025

**Question.** Is this specific to one dataset?

**Result (BRISC, 20 seeds).**

| Method | Balanced accuracy |
|---|---|
| FedAvg | 0.621 ± 0.061 |
| FedRS (α = 0.5) | 0.624 ± 0.062 |
| FedRS (α = 0.25) | 0.636 ± 0.061 |
| Masking (α = 0) | **0.676 ± 0.046** |

Masking exceeded FedRS by 5.2 points, Holm `p < 0.001`, `d = 0.82`, close to the
5.8 points measured on Figshare.

**One observation worth noting.** BRISC is class-balanced and no method collapsed
on it, yet masking still won. The correction therefore does more than prevent
degenerate solutions; it improves accuracy even when nothing is collapsing.

**Caveats.** BRISC has no patient identifiers, so clients are formed by slice,
and it shares source material with Figshare. Both are stated wherever the result
is cited. See [`00_datasets.md`](00_datasets.md).

**Code.** `code/pillar2_federated/stage4_brisc_external.py`

---

## Stage 5: Does the benefit follow local class absence?

**Question.** If the mechanism in Stage 2 is right, the benefit should scale with
how much of the label space is missing locally, and should vanish when nothing is
missing. That is a falsifiable prediction.

**Method.** `α_dir ∈ {0.1, 0.5, 3.0}`, producing `ā ∈ {0.50, 0.31, 0.07}`, each
evaluated at `α ∈ {0, 0.5, 1}` over 8 seeds.

**Result.**

| `α_dir` | `ā` | α = 0 | α = 0.5 | α = 1 | Margin | p |
|---|---|---|---|---|---|---|
| 0.1 | 0.50 | 0.629 | 0.454 | 0.471 | **+17.5** | 0.008 |
| 0.5 | 0.31 | 0.670 | 0.607 | 0.602 | +6.3 | 0.023 |
| 3.0 | 0.07 | 0.659 | 0.657 | 0.656 | +0.2 | 0.836 |

**A correction to an earlier expectation.** Before running this, the expectation
was that the *optimal* `α` would rise as skew decreased. It did not: `α = 0` was
best at every level tested. What changes is the *size* of the advantage, which
falls monotonically with `ā` and becomes indistinguishable from zero when clients
hold almost the full label space. The prediction was right about the direction
of the dependence and wrong about the form, and the manuscript states the
measured behaviour, not the anticipated one.

This is the cleanest single piece of evidence for the mechanism: the correction
switches off exactly where the mechanism says there is nothing to correct.

**Code.** `code/pillar2_federated/stage5_alpha_sweep.py`

---

## Stage 6: Logit calibration, centralised bound, larger backbone

**Question.** Three gaps remained that a reviewer would identify: a missing
comparison against FedLC, no absolute reference point, and results from only one
architecture.

### 6a. Logit calibration (FedLC)

FedLC subtracts a frequency-dependent margin from each logit, which for an absent
class is large enough to be equivalent to masking. Since it generalises FedRS and
is cited in the related work, omitting it empirically would have been the most
likely single cause of rejection.

**Result (20 seeds, Figshare, small backbone).**

| Method | Balanced accuracy |
|---|---|
| FedAvg | 0.584 ± 0.074 |
| FedRS (α = 0.5) | 0.592 ± 0.072 |
| **Logit calibration** | **0.653 ± 0.026** |
| **Masking (α = 0)** | **0.649 ± 0.028** |

Calibration and masking are **statistically indistinguishable** (Δ = 0.004, Holm
`p = 0.123`, 8 of 20 seeds favouring calibration), while both exceed FedAvg by
six to seven points.

**Why this changed the framing of the whole thesis.** An earlier run of the same
comparison had shown calibration ahead by 0.76 points at `p = 0.008`. On repeat
the gap fell to 0.4 points and lost significance. The borderline result did not
survive, which is consistent with the pattern seen throughout this project.

More importantly, the equivalence is itself the finding. Two operators with
different forms, different parameters and different origins produce the same
outcome, while partial scaling produces almost nothing. That places the effect in
the *presence of the absent-class term* rather than in any particular remedy.
The thesis therefore claims a diagnosis, not a method. Masking's practical
advantage is that it introduces no hyperparameter, whereas calibration requires
a temperature.

### 6b. Centralised upper bound

Training on the pooled data with the same backbone and budget gives
**0.705 ± 0.011** over 5 seeds. Masking therefore closes **54%** of the gap
between FedAvg (0.584) and centralised training, which locates the remaining gap
in the federation rather than in the loss function.

### 6c. Residual network backbone

**Result (10 seeds).**

| Method | Three-block CNN | Residual network |
|---|---|---|
| FedAvg | 0.584 | 0.559 |
| FedRS (α = 0.5) | 0.592 | 0.563 |
| Logit calibration | 0.653 | 0.593 |
| Masking | 0.649 | 0.588 |

Two findings, both reported.

The benefit **halves** with the larger backbone, from +6.5 to +2.9 points for
masking. This reproduces on medical images an effect that the FedRS paper
reported on natural images, where a larger feature extractor diluted the
influence of an output-layer correction. It bounds the regime in which the result
should be expected to hold.

The residual network is also **worse in absolute terms** on every objective. With
about 2400 training slices across 17 clients there is too little data per client
to support the larger model. This supports the choice of the small backbone as
the primary configuration, a decision that was made before this experiment and
is now backed by evidence rather than assumed.

**Code.** `code/pillar2_federated/stage6_reviewer_proofing.py`

---

## Consolidated results

Figshare, 18 clients, `α_dir = 0.3`, 20 seeds, patient-level:

| Local objective | Balanced accuracy | Δ vs masking | Holm p |
|---|---|---|---|
| FedAvg | 0.584 ± 0.074 | −0.065 | 0.002 |
| FedRS (α = 0.5) | 0.592 ± 0.072 | −0.058 | 0.002 |
| Logit calibration | 0.653 ± 0.026 | +0.004 | 0.123 |
| Masking (α = 0) | 0.649 ± 0.028 | n/a | n/a |
| *Centralised* | *0.705 ± 0.011* | n/a | n/a |

---

## Implementation choices and their justification

**GroupNorm rather than BatchNorm.** Batch normalisation estimates statistics
from local batches; under non-identical client distributions those estimates
diverge and averaging them is unreliable. Hsieh et al. (2020) showed that group
normalisation recovers most of the accuracy lost to this effect, and later
normalisation benchmarks reproduced the finding. Used in every federated run.

**Aggregation: FedAvg in the main tables, SCAFFOLD checked separately.** The
Stage 1 ablation and the main comparison aggregate by weighted FedAvg; the
restricted loss is the only change to the local objective. Stage 3 (the FedRS
comparison) and the exploratory clustering and C-weighting experiments in
`03_withdrawn_and_null.md` did run with SCAFFOLD control variates. An earlier
version of this document described SCAFFOLD as the optimiser of every run; that
was inaccurate for the ablation and the main comparison and has been corrected.
Real SCAFFOLD (Option II) was later implemented, validated over 20 seeds twice,
and added as an eighth ablation row: it is statistically indistinguishable from
FedAvg aggregation (Δ = +0.25 pp, d = 0.21, p = 0.546 standalone; Δ = +0.43 pp,
d = 0.12, p = 0.230 integrated). Control variates correct drift in the shared
feature extractor, while the restricted loss acts on the output layer, so the
null is consistent with the damage sitting in the output layer. Full record in
[`08_later_experiments.md`](08_later_experiments.md), Stage 13.

**Gradient clipping retained despite hurting alone.** Table in Stage 1 shows
clipping alone is worse than the baseline. It is kept in the final configuration
because it stabilises *in combination* with the restricted loss: once class
masking conditions the loss landscape, clipping bounds residual gradient noise
rather than truncating a still-diverging signal. Stated explicitly in the
manuscript so that the apparent inconsistency is not left for a reviewer to
find.

**20 seeds.** The seed controls the Dirichlet draw, the train/test assignment,
the model initialisation and the batch order, so comparisons are paired and
differences can be tested with a paired test.

**Statistics.** Paired Wilcoxon signed-rank (no normality assumption on the
per-seed differences), Cohen's `d` for paired samples, a 10,000-resample
bootstrap confidence interval, and Holm step-down correction because each method
is compared against three baselines.
