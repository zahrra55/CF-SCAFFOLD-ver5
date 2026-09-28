# CF-SCAFFOLD

Experimental record for a Master's thesis on interpretable brain tumor MRI
classification and federated learning under label skew.

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

### Part I — Interpretable classification from tumor morphology

The starting point proposed by the supervisor: segment the tumor region, extract
descriptors that a radiologist could interpret, and classify tumor type from
those descriptors rather than from a learned representation.

The contribution is a descriptor `C = Ft / Fb`, the ratio of the box-counting
fractal dimension of the isolated tumor to that of the brain tissue with the
tumor removed. It is combined with fifteen further interpretable features
(texture, entropy, lacunarity, shape, edge) and classified with a random forest
under patient-grouped cross-validation.

Documented in [`docs/01_pillar1_classification.md`](docs/01_pillar1_classification.md).

### Part II — Federated learning under missing classes

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
| [`docs/06_references.md`](docs/06_references.md) | 150 references by pipeline stage: link, access route, where and how each is used, how it was verified |
| [`docs/thesis_reference_list.md`](docs/thesis_reference_list.md) | IEEE reference list ready for the thesis end-matter (BibTeX: [`paper/references.bib`](paper/references.bib)) |
| [`docs/07_verification_log.md`](docs/07_verification_log.md) | Audit of every result and reference: what was checked, what was corrected, what is open |
| [`docs/08_later_experiments.md`](docs/08_later_experiments.md) | Stages 9–13: frozen backbone, descriptor ablation, leakage measurement, active contour, SCAFFOLD |

---

## Headline results

Part I, Figshare dataset, patient-grouped five-fold cross-validation:

| Model | Balanced accuracy (patient level) |
|---|---|
| 16 interpretable features + random forest | ~0.92 |
| Reference: Cheng et al. 2015 | 0.913 |

Over ten repeated splits the fractal family adds +1.69 points (p = 0.002); the
ratio C adds nothing on top of Ft and Fb, so it is presented as an interpretable
member of that family rather than a stronger descriptor
([`docs/08`](docs/08_later_experiments.md)).

Part II, Figshare dataset, 18 simulated clients, Dirichlet(0.3), 20 seeds:

| Local objective | Balanced accuracy |
|---|---|
| FedAvg | 0.584 ± 0.074 |
| FedRS (α = 0.5) | 0.592 ± 0.072 |
| Logit calibration | 0.653 ± 0.026 |
| Absent-class masking | 0.649 ± 0.028 |
| Centralised on pooled data | 0.705 ± 0.011 |

These numbers use weighted FedAvg aggregation. Real SCAFFOLD control variates,
added later with the same local objective, are statistically indistinguishable
(+0.25 pp, p = 0.546, 20 seeds).

Full tables, including the external-validation dataset, the partition-severity
sweep, the backbone comparison and the SCAFFOLD check, are in
[`results/tables/`](results/tables/).

---

## Note on scope of the claims

The masking objective evaluated here is the limiting case α → 0 of the scaling
rule published as FedRS. It is not presented as a new operator. The claim is
about which property of these objectives produces the improvement, and about the
regime in which the limiting case is the appropriate choice.

---

## Data availability

Neither dataset is redistributed here. Both are publicly available under their
own licences; see [`docs/00_datasets.md`](docs/00_datasets.md) for sources and
access instructions.

---

## Licence

Code is released under the MIT licence (see `LICENSE`). The datasets retain the
licences of their original publishers.
