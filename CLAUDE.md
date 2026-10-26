# Offer board: notes for Claude Code

This repo tracks open full-time roles for a Carnegie Mellon Materials Science PhD who graduates in May 2027 and holds an Amazon Applied Scientist return offer. The candidate profile, offer line and filters live in `config.yaml`; read it before screening anything.

## What qualifies for the board

A role must meet every rule:

- **Full-time and in the US**, starting in 2027 or with a flexible start date.
- **Aimed at a new PhD.** That means PhD new grad, university grad, early career, or no seniority bar beyond a PhD.
  - Exclude internships, postdocs, fellowships, managers, and staff, principal or lead roles.
  - Senior titles are allowed only at the small AI-for-science companies listed under `filters.science_companies`, and they are marked as stretch roles.
- **Visa-friendly.** Drop roles that require US citizenship, permanent residency or a security clearance, and roles that say they won't sponsor visas.
- **No quant or trading** (firms are listed under `filters.quant_companies`) and **no Amazon roles**.
- **Pay at or near the offer line.** Estimate year-1 total pay at the PhD new-grad level: base, plus typical bonus, plus annualized equity. The midpoint of your estimate must be at least `tiers.near` × `offer_line_k`. Use the posting's own pay range when it has one.
- **Open right now.** Confirm on the employer's own page or applicant-tracking page (Greenhouse, Lever, Ashby, Workday, company careers site), not only on an aggregator.

## How to add or reject roles

Never edit `data/jobs.json` by hand. Write findings to a new file `data/inbox/<YYYY-MM-DD>-<label>.json`, then run `python -m scripts.ingest`. The importer applies the rules above again, removes duplicates, records rejects so they aren't screened twice, and re-renders `README.md` and `docs/index.html`.

```json
{
  "source": "Claude Code sweep",
  "add": [
    {
      "company": "Employer name",
      "role": "Exact job title",
      "url": "https://employer posting URL",
      "location": "City, ST",
      "posted_pay": "pay as stated, or Not listed",
      "est_low_k": 280,
      "est_high_k": 360,
      "fit": 4,
      "track": "science",
      "level": "phd_new_grad",
      "in_us": true,
      "open": true,
      "quant": false,
      "restricted": false,
      "sponsorship": "unclear",
      "posted": "YYYY-MM-DD",
      "why": "One or two sentences tying the role to the candidate's specific experience.",
      "caveat": "One sentence on the main risk."
    }
  ],
  "screened": [
    {"company": "Employer", "role": "Title", "url": "https://...", "reason": "short reason it was rejected"}
  ]
}
```

Field values:
- `track`: `science` for AI-for-science and materials roles, `llm` for LLM and applied ML roles.
- `level`: `phd_new_grad`, `early_career` or `senior`.
- `sponsorship`: `yes` only if the posting says it sponsors, `no` if it says it won't, otherwise `unclear`.
- `fit`: 1–5 against the profile in `config.yaml`.

Put a rejected posting under `screened` only if it looks relevant at first glance; it is shown on the board with its reason. Put clearly irrelevant postings under a third list, `dismissed` (same fields, one-clause reason): they are only remembered so they never come back in `review`. To take a role off the board after reading the posting (for example the automated screen could not read the page and guessed the pay), use a fourth list, `remove`: `[{"url": "...", "reason": "..."}]`.

## Commands

```bash
pip install -r requirements.txt
python -m scripts.update --no-ai   # poll job APIs and trackers, re-check every link, refresh data/state.json "review"
python -m scripts.ingest           # merge data/inbox/*.json and re-render
```

`data/state.json` → `review` lists postings that passed the keyword rules but haven't been screened. Screen them by opening each posting.

## Environment

Job sites are outside the default Trusted network allowlist, so use a cloud environment with **Full** network access. If `scripts.update` prints many `HTTP 0` or `403` warnings, the environment is still on Trusted.

## Git

Commit `config.yaml`, `data/`, `docs/` and `README.md` with a short message such as `Sweep 2026-10-26: 4 new roles`. Push to `main` if you're allowed to; otherwise push your branch and open a pull request. List the new roles, with links, in the PR description or session summary.

## Interview training (`training/`, `scripts/train.py`, `docs/train.html`)

The repo also holds the candidate's daily interview prep. Read `training/README.md` first. The slash command `/train` runs the daily session (today's assignment as a mock interview, then the quiz, then commit); `/mock` runs a timed problem.

- Content: `training/curriculum.json` (tracks and modules), `training/assignments/<id>/` (meta.json, README.md, starter.py, tests.py, solution.py; written design prompts have a rubric in meta.json instead of tests), `training/quiz.json` (flashcards).
- State: `training/progress.json` is an append-only event log plus settings. Never edit it by hand; use the CLI (`today`, `start`, `test`, `done`, `skip`, `quiz --grade`, `goal`, `sync`).
- The page `docs/train.html` is rendered by `python -m scripts.train render` (every mutating command re-renders). It runs tests in the browser for Python and NumPy assignments and exports a `sync` command for progress made there. A copy is published as the private claude.ai artifact "Interview prep, fall 2027" (https://claude.ai/artifact/BzyZR3geDepxXgizeVRDFd; republish it from `python -m scripts.train render --artifact <path>` after content changes), whose `events` collection can be read with the ArtifactData tool and merged with `python -m scripts.train sync <file>`. The board itself is the artifact "Offer board, fall 2027" (https://claude.ai/artifact/L7M5euPp224TP4N9YgjenZ); republish `docs/index.html` there after a sweep.
- Adding content: `python -m scripts.train new <id> --track <track> --module <module> --title "..."`, then fill the files in the style of `training/assignments/lc-two-sum/`, and run `python -m scripts.train selfcheck <id> --starters`. Tests use plain `assert` functions named `test_*`, import from `solution`, must be deterministic and fast, and `numpy`/`python` runtimes may not import torch or scipy (they run in Pyodide).
- Use the Python that has numpy and torch (`/opt/homebrew/Caskroom/miniforge/base/bin/python` on this Mac). Commit `training/`, `docs/` and `README.md` after a session: `Train <date>: <assignment id>`.
