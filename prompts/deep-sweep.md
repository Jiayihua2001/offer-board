# Deep sweep (run in a Claude Code cloud session)

Do a thorough, one-time sweep for open roles that belong on this board. Follow the rules and the inbox format in CLAUDE.md. Today's date is the date of this session.

1. **Refresh the automated sources.** Run `pip install -r requirements.txt` and `python -m scripts.update --no-ai`. If most sources print `HTTP 0` or `403`, stop and say the environment needs Full network access.

2. **Widen coverage.**
   - List at least 60 employers likely to hire this profile at or above the offer line: AI-for-science and materials startups, AI labs, big-tech research groups, and ML-heavy tech companies.
   - For each employer not already in `config.yaml`, find its careers board. If it runs on Greenhouse, Lever, Ashby or Workday, confirm the board name by fetching the public API:
     - `https://boards-api.greenhouse.io/v1/boards/<token>/jobs`
     - `https://api.lever.co/v0/postings/<site>?mode=json`
     - `https://api.ashbyhq.com/posting-api/job-board/<org>`
   - Add each confirmed board to `config.yaml` under `sources`, so later automated runs keep polling it.

3. **Search.**
   - Look for open roles at every employer on the list. Also run the queries under `sources.discovery_queries`, plus any others you think of.
   - Open each candidate on the employer's own site. Confirm it is open and read the pay, location, level, start date and visa terms.

4. **Screen the backlog.** Screen every entry under `review` in `data/state.json`.

5. **Record results.**
   - Write each qualifying role to `data/inbox/<today>-sweep.json` under `add`.
   - Write each relevant rejection under `screened`, with its reason.
   - Run `python -m scripts.ingest` and check the README table.

6. **Commit.** Commit and push following CLAUDE.md. Finish with a short summary that covers:
   - the new roles, with links and one line each on why they fit,
   - the boards you added,
   - anything you couldn't verify.
