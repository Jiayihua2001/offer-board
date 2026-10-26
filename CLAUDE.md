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

Put a rejected posting under `screened` only if it looks relevant at first glance. That stops it from being screened again.

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
