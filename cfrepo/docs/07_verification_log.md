# Verification log

A full audit of the numerical results and the reference list was carried out
before thesis writing began. This document records what was checked, what was
found wrong, and what remains open. It exists so that the state of every claim
is known rather than assumed.

---

## 1. Numerical results — all recomputed from the per-seed data

Every headline figure was recomputed from the stored per-seed values rather than
taken from the printed logs. Means, standard deviations, effect directions,
paired tests and Holm corrections were reproduced independently.

| Claim | Recomputed | Status |
|---|---|---|
| Masking 0.649 vs FedAvg 0.584, $\Delta$ +0.065 | 0.6495 vs 0.5844, +0.0651 | confirmed |
| Masking vs FedRS(0.5), $\Delta$ +0.058 | +0.0580, Holm $p$ = 0.002 | confirmed |
| Masking vs FedLC, not significant | $\Delta$ −0.004, $p$ > 0.10, 8/20 seeds | confirmed |
| Across-seed sd 0.074 → 0.028 | 0.0742 → 0.0283 | confirmed |
| BRISC: masking +5.2 over FedRS | +0.0518, 17/20 seeds | confirmed |
| Centralised bound 0.705 | 0.7054 ± 0.0112 (5 seeds) | confirmed |
| Correction closes 54% of the gap | 53.8% | confirmed |
| ResNet: benefit halves, +6.5 → +2.9 | +0.0651 → +0.0286 | confirmed |
| ResNet worse in absolute terms | 0.593 best vs 0.653 best | confirmed |
| Sweep: +17.5 / +6.3 / +0.2 by absence | matches stored sweep table | confirmed |
| Contrastive arm fails the preregistered test | +0.0126, $p$ = 0.017, 11/16 | confirmed as null |

No numerical claim in either manuscript was contradicted by the audit.

### One claim that was corrected

An earlier draft of the federated manuscript stated that the uncorrected
baseline "converges to a single-class predictor on a quarter of the partitions."
The data do not support this. FedAvg records **zero** runs at or below the
collapse threshold of 0.40 in the main comparison; its worst partition returns
0.447 against a chance level of 0.333. The claim was replaced with a description
of what was actually measured, which is a loss of six to seven points and a
threefold increase in across-seed variance. Collapse at the stated rate occurs
only in the warmup-with-cosine and extended-training configurations of the
ablation, where it is 15 and 14 runs of 20.

### One precision caveat

See [`../results/tables/PRECISION_NOTE.md`](../results/tables/PRECISION_NOTE.md).
The CSVs in this repository were transcribed at three decimal places while the
manuscript statistics were computed at four. Recomputation therefore differs in
the third decimal of some $p$-values without changing any conclusion. The
authoritative CSVs should be downloaded from the Kaggle sessions and substituted.

---

## 2. Reference list — externally verified

Every reference was checked against publisher records. Six contained errors.

| Reference | Error found | Corrected to |
|---|---|---|
| BRISC dataset | Title wrong; volume/article unconfirmed | "BRISC: Annotated Dataset for Brain Tumor **Segmentation and Classification**"; cite by DOI 10.1038/s41597-026-06753-y |
| Sánchez & Martín-Landrove (2022) | Wrong first author | Sánchez is first author, not Martín-Landrove |
| Casella et al. | Cited as arXiv preprint | Published: *IEEE Access* 12:47961–47971, 2024 |
| Battalapalli et al. (2023) | Volume 13; no author list | Volume **14**, art. 1201617, six named authors |
| Guo et al. (FedVLS) | Cited as arXiv preprint | Published: AAAI 2025, vol. 39, pp. 16960–16968 |
| Afshar et al. (2019) | Accuracy given as 90.89% | **90.98%** |

### Two citation-context corrections

**Yagis et al. (2021).** The inflation range is **29 to 55** percentage points,
not 30 to 55. More importantly, that study measured **Alzheimer's and
Parkinson's** cohorts, not brain tumor data. The citation remains appropriate
because the mechanism depends on one subject contributing many adjacent slices,
which holds in the present setting, but both manuscripts now state the original
disease context so that the source is not misrepresented.

**BRISC independence.** The dataset publication states that its images were
collated from existing public datasets, and its documentation lists SARTAJ and
Br35H, the same collections that combine with the Cheng data in the Nickparvar
aggregate. BRISC is therefore **not independent** of the primary dataset. All
wording implying independence has been removed; the experiment is described as a
robustness check.

---

## 3. Withdrawn attribution

An earlier draft attributed to the FedRS paper the observation that a larger
feature extractor dilutes the effect of output-layer corrections. That statement
could not be confirmed in the source. It has been removed from both manuscripts
and replaced with the direct argument, which does not require the citation: a
correction acting on the output layer has less to act on when more of the
classification is carried by the feature extractor, and the measurement in this
work stands on its own.

Two related items remain **unverified** and are listed in section 5 below.

---

## 4. Internal consistency

**Two baselines with different values.** The ablation table reports a federated
baseline at 0.551 and the main comparison reports FedAvg at 0.584. Both are
correct: the ablation was run at twenty rounds with two local epochs, the main
comparison at forty rounds with one. The manuscripts now state this where the
tables appear, so the difference is explained rather than left to be discovered.

**Cross-document agreement.** The means and standard deviations quoted in the
manuscripts, in `docs/02_pillar2_federated.md` and in `results/tables/` agree to
three decimal places throughout.

---

## 5. What the FedRS paper already contains

The full text of Li and Zhan (2021) was obtained and read, resolving the largest
open question in this project. Three things previously treated as contributions
of the present work are already in that paper, and the manuscript now says so
explicitly in a dedicated subsection of the related work.

| Item | Status in FedRS |
|---|---|
| The value $\alpha = 0$ | Evaluated. Their tested set is $\{0.0, 0.1, 0.5, 0.9, 1.0\}$ |
| Analysis of $\alpha = 0$ | Present. Described as degenerating to "fixed proxies" that receive no update and do not contribute to aggregation |
| Empirical verdict on $\alpha = 0$ | "comparable results with $\alpha = 0.5$ in most cases, while it could become worse with more local epochs" |
| Pairing with SCAFFOLD | Present as "Scaffold/RS", and it is their best configuration |
| Backbone size affects the benefit | Present in Appendix D, as a conjecture: weaker gains with VGG11 than with the smaller network |

Two qualifications matter for how this is reported. First, $\alpha = 0$ appears
in their convergence figures but not in their headline tables, and the setting
they carry forward is $\alpha = 0.5$. Second, the backbone observation is phrased
as a conjecture ("we guess"), not as a controlled ablation.

**What this leaves as genuinely ours.** The regime is different: their evaluation
uses ten to one hundred classes with two to twenty per client, ours has three
classes with mean local absence 0.31. In that regime $\alpha = 0$ is not
comparable to $\alpha = 0.5$ but ahead of it by 5.8 points. The sweep in Stage 5
explains the discrepancy rather than disputing it, by showing the benefit to be a
monotone function of local absence, so their finding and ours are two points on
one curve. Their own caveat supports this: the case they found unfavourable is
many local epochs, and we use one. To that we add the comparison against logit
calibration, which neither originating paper reports, the centralised reference
point, and patient-level evaluation on medical data.

**Actions taken.** The manuscript's related work now contains a subsection
setting out the overlap in full; the contributions list and the limitations
section were rewritten to match; and the attribution of the backbone observation
was restored, phrased as the conjecture it is.

---

## 6. Open items — to be resolved before submission

1. **Precision substitution.** Replace the transcribed CSVs with the originals
   from Kaggle, then confirm the manuscript statistics against them. See
   [`../results/tables/PRECISION_NOTE.md`](../results/tables/PRECISION_NOTE.md).

2. **Preprints.** Zhao et al. (2018) and Wu et al. (Motley, 2022) remain arXiv
   preprints with no peer-reviewed venue. Cite them as preprints.

### Resolved since the first audit

- **FedRS content.** Resolved in full; see section 5 above.
- **BRISC record.** Verified as *Scientific Data*, **volume 13, article 361,
  published 5 February 2026**. The earlier "volume 12, article 845, 2025" was a
  different paper appearing inside BRISC's own reference list. The author list
  has seven names and includes Mansoor Fateh; Yasin Rezvani and Sadjad Rezvani
  are distinct authors.
- **Figshare version.** Version 5, posted 2 April 2017, is the release used and
  the one cited in the literature.
- **A DOI to avoid.** `10.1371/journal.pone.0139642` is *not* a Cheng paper. It
  is "Assessing the Power of Exome Chips", an unrelated genetics study that
  happens to sit in the same PLOS ONE issue. The correct DOI for the brain tumor
  classification paper is `10.1371/journal.pone.0140381`. The dataset page cites
  that paper together with the 2016 retrieval paper, `10.1371/journal.pone.0157112`.
