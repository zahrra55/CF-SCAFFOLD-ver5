# Note on the precision of these tables

**Read this before quoting a $p$-value from these files.**

The per-seed accuracies in this directory were transcribed from the printed run
logs, which display three decimal places. The scripts themselves write four
decimal places to `/kaggle/working/*.csv` during the run, and the statistics
reported in the manuscript were computed by the scripts on those four-decimal
values.

The consequence is that recomputing a test from the files here reproduces the
manuscript to within a small margin but not exactly:

| Quantity | Manuscript (4 dp source) | Recomputed here (3 dp) |
|---|---|---|
| Masking vs FedLC, Holm $p$ | 0.123 | 0.108 |
| Masking vs FedRS on BRISC, Holm $p$ | 0.0009 | 0.0015 |
| Masking vs FedAvg, Cohen's $d$ | 0.91 | 0.90 |

No conclusion changes: the two significant comparisons remain significant at
every reasonable threshold, and the non-significant one remains non-significant.
The means, standard deviations and effect directions agree to three decimals
throughout.

## What to do about it

The authoritative files are the CSVs written by the scripts during each Kaggle
run, at these paths:

| Experiment | Path in the Kaggle session |
|---|---|
| Stage 1 ablation | `/kaggle/working/stability_log.csv` |
| Stage 3 FedRS comparison | `/kaggle/working/fedrs_compare_log.csv` |
| Stage 4 BRISC | `/kaggle/working/brisc_fedrs_log.csv` |
| Stage 5 alpha sweep | `/kaggle/working/alpha_sweep_log.csv` |
| Stage 6 FedLC, centralised, ResNet | `/kaggle/working/stage8_log.csv` |
| Stage 7 contrastive | `/kaggle/working/bio_contrastive_log.csv` |

Download those from the notebook output and replace the corresponding files
here. After replacing them, recompute the tables and confirm that the values in
the manuscript match; if any differs, the manuscript is what should change,
since the CSV is the measurement and the manuscript is the report of it.

Until that replacement is made, treat the files in this directory as accurate to
three decimal places and cite the manuscript for exact test statistics.
