# CF-SCAFFOLD

Experimental record for the Master's thesis
**Fractal-Chaos Guided Federated Learning for Brain Tumor MRI Classification**.

**For the examining committee:** [`COMMITTEE_GUIDE.md`](COMMITTEE_GUIDE.md) lists
which script reproduces each result of the thesis, in which order to run them,
the hardware and time each needs, and the limits that apply to each claim.

**Researcher:** Zahraa Ibrahim Khaleel
**Supervisor:** Prof. Dr. Mohammed Sahib Mehdi
**Department of Computer Science, College of Science, Al-Nahrain University, Baghdad**

---

## Purpose of this repository

This repository is the complete experimental record of the thesis. It exists so
that every reported number can be traced to the script that produced it, the
dataset it was computed on, and the decision that motivated running it. It
includes results that did not work and claims that were withdrawn, because those
shaped the final method as much as the positive findings did.

Anyone reading the thesis should be able to open this repository and answer three
questions for any figure or table: which script produced it, on which data, and
why that experiment was run at all.

---

## The two parts of the work

The thesis contains two methodologically separate investigations. They share a
dataset and a preprocessing pipeline but answer different questions, and neither
depends on the other.

### Part I: Interpretable classification from tumor morphology

The starting point proposed by the supervisor: segment the tumor region, extract
descriptors that a radiologist could interpret, and classify tumor type from
those descriptors rather than from a learned representation.

The contribution is a descriptor `C = Ft / Fb`, the ratio of the box-counting
fractal dimension of the isolated tumor to that of the brain tissue with the
tumor removed. It is combined with fifteen further interpretable features
(texture, entropy, lacunarity, shape, edge) and classified with a random forest
under patient-grouped cross-validation.

Documented in [`docs/01_pillar1_classification.md`](docs/01_pillar1_classification.md).

### Part II: Federated learning under missing classes

When the same task is distributed across simulated hospitals, most sites hold
fewer than all three tumor types. Training then collapses. This part traces the
collapse to the gradient on absent-class logits and evaluates corrections that
remove that term.

Documented in [`docs/02_pillar2_federated.md`](docs/02_pillar2_federated.md).

---

## Repository layout

```
docs/       stage-by-stage record, one document per part plus supporting material
code/       the scripts that produced every reported number
            (code/fractal_chaos_guided_fl/ holds the final stages 17 to 19)
tools/      scripts that built and verified the reference list
results/    tables of the actual measured values, and the figures built from them
paper/      the manuscripts prepared from these results
webapp/     an interactive presentation of the results for a non-programming reader
```

The application in `webapp/` reads the stored results and presents them; it does
not retrain. See [`webapp/README.md`](webapp/README.md) for how to run and
deploy it.

---

## Documentation index

| Document | Contents |
|---|---|
| [`docs/00_datasets.md`](docs/00_datasets.md) | Datasets, licences, access, preprocessing |
| [`docs/01_pillar1_classification.md`](docs/01_pillar1_classification.md) | Segmentation, descriptors, classification, results |
| [`docs/02_pillar2_federated.md`](docs/02_pillar2_federated.md) | Federated experiments, stage by stage |
| [`docs/03_withdrawn_and_null.md`](docs/03_withdrawn_and_null.md) | Results retracted or reported as null, with reasons |
| [`docs/04_decision_log.md`](docs/04_decision_log.md) | Why each method, metric, dataset and value was chosen |
| [`docs/05_reproducibility.md`](docs/05_reproducibility.md) | Environment, seeds, how to re-run each experiment |
| [`docs/06_references.md`](docs/06_references.md) | 166 references by pipeline stage: link, access route, where and how each is used, how it was verified |
| [`docs/thesis_reference_list.md`](docs/thesis_reference_list.md) | IEEE reference list ready for the thesis end-matter (BibTeX: [`paper/references.bib`](paper/references.bib)) |
| [`docs/07_verification_log.md`](docs/07_verification_log.md) | Audit of every result and reference: what was checked, what was corrected, what is open |
| [`docs/08_later_experiments.md`](docs/08_later_experiments.md) | Stages 9 to 19: frozen backbone, descriptor ablation, leakage, active contour, SCAFFOLD, shortcut analyses, radiologist-free classification, fractal-chaos descriptors, matched heterogeneity damage, final extension |

---

## Headline results

Final descriptor model (Stage 19), Figshare, 218 patients, ten repeated
patient-grouped splits, patient-level balanced accuracy:

| Configuration | Balanced accuracy |
|---|---|
<<<<<<< HEAD
| Final set: 21 descriptors (fractal-chaos and others) | 0.948 ± 0.009 |
| Tumour-only descriptors (robust to the dataset shortcut) | 0.877 ± 0.008 |
| Fractal-chaos family alone (fractal dimensions, lacunarities, fuzzy entropy) | 0.851 ± 0.008 |
=======
| 17 interpretable features + random forest, radiologist masks | 0.897 |
| Same model, tumour located by a small U-Net (no radiologist at test time) | 0.892 |
| 13 tumour-region features only (least dependent on the dataset shortcut) | 0.813 |

An earlier version of this table reported about 0.92 and placed the model above
Cheng et al. (2015, 0.913). That figure came from feature code with four defects
(entropy, texture, box height, a missing Ft) that were corrected in the final
pipeline; the corrected model is below that benchmark, and the claim has been
withdrawn. The contribution of Part I is not raw accuracy but an interpretable
classifier that can run without a radiologist and whose behaviour on a dataset
with a documented shortcut (Wallis and Buvat, 2022) has been measured
([`docs/08`](docs/08_later_experiments.md), Stages 14 to 16).
>>>>>>> 16b98a9120b23114c48156d2f63e733673cf6f5c

The final set was selected on the same data by a preregistered rule, so 0.948 is
slightly optimistic; the tumour-only figure is the estimate robust to the
slice-selection shortcut documented by Wallis and Buvat (2022). An earlier figure of
about 0.92 and the comparison with Cheng et al. (2015) were withdrawn after a
feature-code audit (see [`docs/07`](docs/07_verification_log.md)).

Glioma versus non-glioma, a proxy for malignant versus predominantly benign
tumours: 0.989 with radiologist masks and about 0.96 after automatic segmentation
by a U-Net (0.966 and 0.962 in two runs), against 0.778 for brain geometry alone.

Fractal-chaos guided federated learning, 18 simulated hospitals, Dirichlet(0.3),
20 seeds, identical hospitals and test patients (Stages 18 and 19):

| Model | Centralised | Federated | Damage from heterogeneity |
|---|---|---|---|
| CNN on images | 0.736 | 0.584 | 15.2 points |
| MLP on the final descriptor set | 0.917 | 0.900 (patient level 0.953) | 1.7 points |

The damage gap grows with heterogeneity (20.6 points at alpha 0.1, 5.6 at alpha 1.0).
The fractal-chaos family alone also resists heterogeneity, but the non-fractal
descriptors resist it at least as well, so robustness is a property of fixed
descriptors in general.

Federated learning on images, local objectives compared over 20 seeds:

| Local objective | Balanced accuracy |
|---|---|
| FedAvg | 0.584 ± 0.074 |
| FedRS (α = 0.5) | 0.592 ± 0.072 |
| Logit calibration | 0.653 ± 0.026 |
| Absent-class masking | 0.649 ± 0.028 |
| Centralised on pooled data | 0.705 ± 0.011 |

Real SCAFFOLD control variates are statistically indistinguishable from FedAvg
aggregation with the same objective (+0.25 pp, p = 0.546). All tables are in
[`results/tables/`](results/tables/).

---

## Note on scope of the claims

The masking objective evaluated here is the limiting case α → 0 of the scaling
rule published as FedRS. It is not presented as a new operator. The claim is
about which property of these objectives produces the improvement, and about the
regime in which the limiting case is the appropriate choice.

---

## Data availability

No dataset is redistributed here (Figshare brain tumour dataset, BRISC, and the
BraTS 2020 training set used for grading). Both are publicly available under their
own licences; see [`docs/00_datasets.md`](docs/00_datasets.md) for sources and
access instructions.

---

## Licence

Code is released under the MIT licence (see `LICENSE`). The datasets retain the
licences of their original publishers.
