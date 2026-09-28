# CF-SCAFFOLD — interactive companion

A small web application that presents the thesis results to a reader who does
not program. It runs on a laptop during the viva and can also be published as a
link for the supervisor and the committee.

---

## Running it on your machine

```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Nothing else is needed; there
is no database, no server to configure and no account to create.

To stop it, press `Ctrl+C` in the terminal.

---

## Publishing it as a link

The application is designed for **Streamlit Community Cloud**, which is free
and deploys directly from a GitHub repository.

1. Put this `webapp` folder inside your repository and push it.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Choose the repository, set the main file path to `webapp/app.py`, deploy.

The result is a permanent link. The same code runs locally and online without
modification, so there is only one version to maintain and to explain.

---

## What each file does

```
app.py                  the interface: three pages, all display logic
lib/descriptor.py       the scientific core: how C = Ft/Fb is computed
data/*.csv              the measured results, one file per experiment
requirements.txt        the packages needed
.streamlit/config.toml  colours and fonts
```

The separation between `app.py` and `lib/descriptor.py` is deliberate. The
descriptor is the contribution and can be read, checked and defended on its own,
without anyone needing to look at interface code. If a committee member asks how
C is computed, the answer is one file of about 200 lines with a comment on every
step.

---

## The one design decision worth explaining

**The application does not train anything.**

A single federated run takes about 130 seconds on a GPU, and the full comparison
across four methods and twenty partitions takes roughly 45 minutes. A browser
request times out long before that, and the machine running the application has
no GPU.

So the architecture is: experiments run on Kaggle, results are stored as CSV
files, and the application reads and presents them. The only live computation is
the fractal descriptor on page 3, which takes a few seconds and needs no GPU.

This is also better for a viva. The committee sees the results immediately
instead of watching a progress bar.

If asked *"is this showing real results or a simulation?"* — the numbers in
`data/` are the measured per-seed values from the Kaggle runs, the same values
that produced every table in the manuscript. The partition simulator on page 1
is a live illustration of the Dirichlet draw, and it is labelled as such.

---

## The three pages

**1. The problem.** Explains why federated training fails when sites lack tumor
types, and lets the reader move a slider to see it happen. Moving the
concentration from 0.3 to 3.0 shows sites going from missing classes to holding
all of them, which is the condition the rest of the work depends on.

**2. The results.** The measured comparison, the statistical tests, the
dependence of the benefit on how much is missing, and the two robustness checks.
The page states plainly that masking and logit calibration cannot be
distinguished, because that equivalence is the finding rather than an
inconvenience.

**3. The descriptor.** Computes C on a supplied or generated slice and shows
every intermediate step: the three regions, the two box-counting fits, and the
ratio. It closes by reporting how unevenly C separates the three tumor types on
its own, including the weak result for meningioma.

---

## Preparing to defend it

Read `lib/descriptor.py` first. It is the part with scientific content, and
every function has a docstring saying not only what it does but why it is done
that way: why the background is removed before counting, why intensities are
rescaled inside each region, why small regions are rejected.

Then read `app.py` from the top. It runs from top to bottom like an ordinary
script; there are no routes, no callbacks and no separate template files. Each
page is one `if` branch.

Questions you should be able to answer, and where the answer is:

| Question | Where |
|---|---|
| Why a ratio rather than the tumor dimension alone? | `descriptor.py`, docstring of `tumor_to_brain_ratio` |
| Why remove the background first? | `descriptor.py`, docstring of `brain_mask` |
| Why reject small regions? | `descriptor.py`, the two rejection branches |
| Why not train in the browser? | this file, "the one design decision" |
| Are these real results? | `data/`, and the note above |
| Why is C weak for meningioma? | page 3, closing section |

---

## What this application is not

It is not a diagnostic tool, and it should not be described as one to anyone.
It presents a research result. The generated example on page 3 is synthetic and
carries no clinical meaning.
