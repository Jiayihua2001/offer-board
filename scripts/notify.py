"""Open a GitHub issue summarizing new and closed roles, so GitHub emails you."""
from __future__ import annotations

import os

import requests

from .render import tier


def post_issue(new_jobs: list[dict], closed: list[dict], cfg: dict, run_date: str, log) -> None:
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not (cfg.get("notify", {}).get("github_issue") and token and repo):
        return
    if not new_jobs and not closed:
        return
    lines = []
    if new_jobs:
        lines += [f"## {len(new_jobs)} new role{'s' if len(new_jobs) != 1 else ''}", "",
                  "| Company | Role | Est. year-1 pay | vs offer | Fit | H-1B |", "|---|---|---|---|---|---|"]
        for j in new_jobs:
            _, d = tier(j, cfg)
            lines.append(f"| {j['company']} | [{j['role']}]({j['url']}) | ${j['est'][0]}K–${j['est'][1]}K | "
                         f"{'+' if d > 0 else ''}{d}% | {j.get('fit', '')}/5 | {(j.get('visa') or ['ask'])[0]} |")
        lines.append("")
        for j in new_jobs:
            lines.append(f"- **{j['company']}, {j['role']}**: {j.get('why', '')} _Watch for:_ {j.get('caveat', '')}")
        lines.append("")
    if closed:
        lines += ["## Closed and removed", ""] + [f"- {c['name']}" for c in closed] + [""]
    title = (f"{len(new_jobs)} new role{'s' if len(new_jobs) != 1 else ''}" if new_jobs else "Board update") + \
            (f", {len(closed)} closed" if closed else "") + f" ({run_date})"
    try:
        r = requests.post(f"https://api.github.com/repos/{repo}/issues",
                          headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                          json={"title": title, "body": "\n".join(lines)}, timeout=30)
        if r.status_code >= 300:
            log.warn(f"could not open issue: HTTP {r.status_code} {r.text[:200]}")
    except requests.RequestException as e:
        log.warn(f"could not open issue: {e!r}")
