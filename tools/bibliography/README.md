# Bibliography verification tools

These scripts built and checked `docs/06_references.md`, `paper/references.bib`
and `docs/references_verification.csv`. Re-run them before submission to catch
any link that has moved.

| File | Role |
|---|---|
| `entries.py` | Master list: key, pipeline stage, identifier (DOI, arXiv, URL, ISBN), expected title, year, and where the reference is used |
| `verify.py` | Looks up every identifier in Crossref, DataCite, the arXiv API, Open Library or the page itself; checks title and year; resolves each DOI through doi.org; records open-access status from OpenAlex. Writes `verified.json` |
| `format.py` | Builds IEEE strings and BibTeX from the verified metadata. The BibTeX is ASCII-only (LaTeX escapes) and applies the IEEE rule "more than six authors: first author et al." |
| `make_docs.py` | Writes the annotated list, the end-matter list and the CSV |
| `tex_escape.py` | Accent-to-LaTeX conversion; raises an error on any unmapped non-ASCII character |

```bash
cd tools/bibliography
python3 verify.py      # network access required; about 2 minutes
python3 format.py
python3 make_docs.py
```

Paths inside the scripts point to the working directory used when they were
written (`/home/claude/refs`); edit the first lines of each script to your own
folder before re-running. Some publishers (OUP, ACM, MDPI, Elsevier, Wiley)
return HTTP 403 to any script; those links open normally in a browser.
