# How to publish this repository

This file is for you, not for readers. Delete it after the first push.

---

## Step 1 — create the empty repository on GitHub

Go to https://github.com/new and set:

- **Owner:** `zahrra55`
- **Repository name:** `CF-SCAFFOLD`
- **Description:** Experimental record for a Master's thesis on interpretable brain tumor MRI classification and federated learning under label skew
- **Visibility:** your choice. Public makes the repository citable in the thesis
  and the paper; private keeps it closed until the viva. You can switch a private
  repository to public later without losing anything.
- **Do not** tick "Add a README", "Add .gitignore" or "Choose a licence". This
  archive already contains all three, and ticking them creates a conflict on the
  first push.

---

## Step 2 — push from your machine

Download and unzip this archive, open a terminal in the unzipped folder, then:

```bash
git init
git add .
git commit -m "Experimental record: datasets, code, results, documentation"
git branch -M main
git remote add origin https://github.com/zahrra55/CF-SCAFFOLD.git
git push -u origin main
```

If GitHub asks for a password, it wants a **personal access token**, not your
account password. Create one at Settings → Developer settings → Personal access
tokens → Tokens (classic), with the `repo` scope, and paste it when prompted.

---

## Step 3 — after the push

1. Delete this file and commit the deletion:
   ```bash
   git rm PUSH_INSTRUCTIONS.md
   git commit -m "Remove setup instructions"
   git push
   ```

2. Check that `README.md` renders correctly on the repository home page and that
   the links in the documentation index all open.

3. Cite the repository in the thesis and in the paper's data availability
   statement, for example:

   > Code and experimental records are available at
   > https://github.com/zahrra55/CF-SCAFFOLD

---

## Before you push, check these three things

**Your email address.** `paper/manuscript.tex` still contains the placeholder
`[YOUR EMAIL]`. Fill it in or remove the line.

**Author order.** Confirm with your supervisor that the author order in the
manuscript is what he expects.

**No data files.** The `.gitignore` excludes `.mat`, `.npz` and `.pt` files, so
no dataset content should be committed. Confirm with `git status` before the
first commit that only code, documentation, figures and CSV tables are staged.
Redistributing the datasets would breach their licence terms and would make the
repository very large.

---

## Keeping it current

The repository should be updated when results change, not only at the end. After
any run that changes a reported number:

1. Update the CSV in `results/tables/`.
2. Update the corresponding table in `docs/`.
3. Commit with a message that says what changed and why, for example
   `Rerun FedLC comparison: gap falls to 0.4pp, no longer significant`.

A commit history that records when each number was measured is itself evidence
that the work was done as described.
