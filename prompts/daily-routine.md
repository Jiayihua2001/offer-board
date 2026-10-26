# Daily update (for a Claude Code routine, no API key needed)

Keep the offer board current. Follow the rules and the inbox format in CLAUDE.md. Today's date is the date of this run.

1. **Refresh the sources.** Run `pip install -r requirements.txt` and `python -m scripts.update --no-ai`. This merges any inbox files, polls the job APIs and trackers, re-checks every link, and lists unscreened postings under `review` in `data/state.json`.

2. **Screen the backlog.** Screen up to 40 entries from `review`. Open each posting on the employer's site and apply the rules.

3. **Search for new roles.**
   - Run five of the queries under `sources.discovery_queries`. Rotate through them by day of the year so every query gets a turn.
   - Check each promising result on the employer's site.

4. **Record results.**
   - Write qualifying roles to `data/inbox/<today>-routine.json` under `add`.
   - Write relevant rejections under `screened`.
   - Run `python -m scripts.ingest`.

5. **Commit.** Commit and push to `main` with the message `Routine update <today>`. If any roles were added, list them with links in your final message.
