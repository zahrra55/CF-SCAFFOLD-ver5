# Verification log

A full audit of the numerical results and the reference list was carried out
before thesis writing began. This document records what was checked, what was
found wrong, and what remains open. It exists so that the state of every claim
is known rather than assumed.

---

## 1. Numerical results: all recomputed from the per-seed data

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

## 2. Reference list: externally verified

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

## 6. Bibliography rebuilt and re-verified (September 2026)

The 32-item list was replaced by 150 references organised by pipeline stage.
Every entry was checked against the registries themselves, not against
summaries or earlier drafts.

| Identifier | Count | Check |
|---|---|---|
| DOI | 108 | Crossref or DataCite record; title and year matched; resolved through doi.org |
| arXiv | 21 | arXiv API; 13 have peer-reviewed versions confirmed on OpenReview or in the NeurIPS/MLSys proceedings and are cited in that form; 8 remain preprints |
| Proceedings or journal page, no DOI | 16 | Official PMLR, NeurIPS, MLSys, JMLR page fetched; authors and pages read from its citation meta-tags |
| Book | 6 | Open Library (ISBN) or Crossref (book DOI) |
| JSTOR only | 1 | Holm (1979): no DOI exists; JSTOR 4615733 and MR538597 |

Result: 150/150 verified. `paper/references.bib` compiles with IEEEtran with
zero BibTeX warnings and zero LaTeX errors. Of the 120 non-DOI and free
full-text links, 103 returned HTTP 200 to a script; the remaining 17 are
publishers that block automated clients (OUP, ACM, MDPI, Elsevier, Wiley, Cell),
a figshare page returning 202, and one Harvard repository rate limit.

**Corrections found.**

- Kornblith et al., CVPR 2019: pages 2656–2666 (IEEE Xplore, the DOI version), not 2661–2671 (CVF pagination).
- Islam et al., *Neural Processing Letters*: vol. 55, no. 4, pp. 3779–3809, issue year 2023.
- MOON, CVPR 2021: pp. 10708–10717. Class-balanced loss, CVPR 2019: pp. 9260–9269.
- Gonzato et al.: the sampling-bias paper is 1998 (with Marzocchi); the 2000 paper has Ciccotti as third author. Both now listed.
- Liu, *J. Biomed. Inform.* 2009: DOI 10.1016/j.jbi.2009.07.003 (the .008 DOI is a different paper).
- *Pattern Recognition* 2024 FL survey: first author H. Guan, not "Rahman".
- arXiv:2502.11323 is a statistical theory of overfitting for imbalanced classification (Lyu, Zhou, Zhong); its described use was corrected.
- Mandelbrot (1982): San Francisco, W. H. Freeman.
- Three free-copy links suggested by OpenAlex were unusable (Litjens: an image file; ResNet: a dead unofficial mirror; Breiman: an anonymous short link) and were replaced by the arXiv copy, the CVF open-access page, and the author's Berkeley PDF, each checked.
- No peer-reviewed "brain-tumour-specific leakage paper" exists; Kapoor and Narayanan (2023) is the companion to Yagis et al. (2021).

Tools and raw results: `tools/bibliography/`, `docs/references_verification.csv`.

---

## 7. SCAFFOLD naming vs implementation: resolved

A technical review found that the canonical ablation code, although the project
is named CF-SCAFFOLD, aggregated by weighted FedAvg without control variates,
and that `02_pillar2_federated.md` and `04_decision_log.md` described SCAFFOLD as
the optimiser of every run. An audit of all federated scripts established the
exact scope: the Stage 1 ablation and the main comparison used FedAvg; Stage 3
(FedRS comparison) and the exploratory clustering and C-weighting experiments
used real control variates.

Real SCAFFOLD (Option II) was implemented, checked against four silent-failure
modes, validated over 20 seeds against a preregistered criterion, and integrated
as an additive eighth ablation row whose default-off path is bit-identical to the
original code. It is statistically indistinguishable from FedAvg aggregation.
Both documents were corrected. Record: `08_later_experiments.md`, Stage 13.

---

## 9. Feature-code audit, Part I figures, and novelty checks (October 2026)

**Feature defects corrected.** Entropy computed from a density instead of
probabilities; texture computed on the whole image with a zero background; box
height s instead of s·G/M; Ft missing from the feature set; ties in the patient
vote always resolved to class 0; lacunarity skipping the last row and column of
boxes; the SCAFFOLD global control variate sample-weighted and scaled by 17/18
instead of the plain mean of the paper. Each correction was checked on a test the
old code fails. Federated rows 1 to 7 were verified bit-identical to the
documented code.

**Figures superseded.** Part I patient-level balanced accuracy 0.919 becomes
0.897; the statement that Part I exceeds Cheng et al. (2015) is withdrawn. The
0.2-point cost of automatic masks is restated with its interval, -0.6 points
(95% CI -4.6 to +3.7). The share of clients missing a tumour type is reported as
12.6 of 15.2 on average over 20 seeds, replacing values from single partitions.

**Novelty checks.** The brain-geometry shortcut was first shown by Wallis and
Buvat (2022) and is presented as a replication. A second search, including the
papers citing Wallis and Buvat (mainly through Semantic Scholar; Google Scholar
could not be queried), found no earlier publication of the stratified comparison
of feature families (Stage 15), of the area analysis of the fractal descriptors
(Stage 14B), or of the radiologist-free pipeline on this dataset (Stage 16). The
feature-level pattern of Stage 16 is consistent with published work on radiomic
stability and is cited as such. Run a manual Google Scholar check before
submission.

**References.** Sixteen references were added for these analyses and verified
against the registries; the list now has 166 entries and compiles with zero
BibTeX warnings.

---

<<<<<<< HEAD
## 11. Code audit before submission (October 2026)

All ten scripts that produced final results passed static analysis with no undefined names or conflicting definitions, and all ran on the real data. A logic review found two minor errors, both in the federated arm, and both were corrected in Stage 19 version 1.1. Patient-level votes broke ties towards the lowest class index because the vote received one-hot predictions instead of model probabilities, which is the same defect as F5; the corrected patient-level figure is 0.953 (0.951 before). Missing values were imputed with a median of pooled training data, which a federation cannot compute; they are now imputed with the mean of observed values, which hospitals can obtain from aggregated sums and counts. The Stage 17 patient-level federated figure is superseded by Stage 19.

Three methodological points are documented rather than corrected: the FuzEn2D seed-stability criterion was uninformative, the final descriptor set was selected on the same data that report it, and the post-segmentation figure comes from a single split whose value varied by about 0.4 points between two runs because U-Net training on GPU is not deterministic.

---

## 12. Open items: to be resolved before submission
=======
## 10. Open items: to be resolved before submission
>>>>>>> 16b98a9120b23114c48156d2f63e733673cf6f5c

1. **Precision substitution.** Replace the transcribed CSVs with the originals
   from Kaggle, then confirm the manuscript statistics against them. See
   [`../results/tables/PRECISION_NOTE.md`](../results/tables/PRECISION_NOTE.md).

2. **Preprints.** Eight references have no peer-reviewed version and are cited
   as preprints: Hsu et al. (2019), Zhao et al. (2018), Wu et al. (Motley, 2022),
   Goyal et al. (2017), Colas et al. (2018), Park et al. (2025), Haseeb et al.
   (2025), Lyu et al. (2025). Re-check the three 2025 items before submission;
   they may have been accepted somewhere by then.

3. **Stage 12 symmetry seed on real data.** Only a synthetic check is recorded.
   Run it on the Figshare data before making any claim about it.

4. **Stage 13 per-seed log of the integrated eighth row.** The summary is in
   `results/tables/ablation_with_scaffold_summary.csv`; export the Kaggle
   `stability_log.csv` to replace it with per-seed values.

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
