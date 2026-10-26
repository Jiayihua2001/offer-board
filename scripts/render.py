"""Render README.md (between markers) and docs/index.html from the board data."""
from __future__ import annotations

import datetime as dt
import json

from .common import ROOT

START, END = "<!-- BOARD:START -->", "<!-- BOARD:END -->"
VISA_WORD = {"yes": "sponsors", "stated": "says it sponsors", "ask": "**ask first**"}


def tier(job: dict, cfg: dict) -> tuple[str, int]:
    mid = (job["est"][0] + job["est"][1]) / 2
    ratio = mid / float(cfg["offer_line_k"])
    t = "above" if ratio >= cfg["tiers"]["above"] else ("near" if ratio >= cfg["tiers"]["near"] else "below")
    return t, round((ratio - 1) * 100)


def _md(s: str) -> str:
    return str(s or "").replace("|", "\\|").replace("\n", " ")


def _row(j: dict, cfg: dict) -> str:
    t, d = tier(j, cfg)
    delta = f"{'+' if d > 0 else ''}{d}%"
    tags = []
    if j.get("added") and j["added"] >= _new_since():
        tags.append("**New**")
    if j.get("stretch"):
        tags.append("stretch")
    if j.get("stale"):
        tags.append("not confirmed open lately")
    visa = VISA_WORD.get((j.get("visa") or ["ask"])[0], "ask first")
    return (f"| {_md(j['company'])} | [{_md(j['role'])}]({j['url']}){' ' + ', '.join(tags) if tags else ''} | "
            f"{_md(j['location'])} | ${j['est'][0]}K–${j['est'][1]}K ({delta}) | {'●' * int(j.get('fit', 0))} | "
            f"{visa} | {_md(j.get('posted', ''))} |")


def _new_since() -> str:
    return (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=3)).isoformat()


def sort_jobs(jobs: list[dict]) -> list[dict]:
    def key(j):
        mid = (j["est"][0] + j["est"][1]) / 2
        return (-(j.get("fit", 0) - (1 if j.get("stretch") else 0)), -mid)
    return sorted(jobs, key=key)


def readme_block(jobs, screened, review, closed_recent, state, cfg) -> str:
    above = [j for j in jobs if tier(j, cfg)[0] == "above"]
    near = [j for j in jobs if tier(j, cfg)[0] == "near"]
    head = ("| Company | Role | Location | Est. year-1 pay (vs offer) | Fit | H-1B | Posted |\n"
            "|---|---|---|---|---|---|---|")
    lines = [START, "",
             f"**{len(jobs)} open roles** · {len(above)} above your ${cfg['offer_line_k']}K line · {len(near)} near it · "
             f"last run {state.get('last_run', '')} · AI spend this month ${state.get('spend', {}).get(dt.datetime.now(dt.timezone.utc).strftime('%Y-%m'), 0):.2f}"
             f" of ${cfg['ai']['monthly_budget_usd']}", ""]
    for label, group in (("Above your offer line", above), ("Near your offer line", near)):
        lines += [f"### {label}", "", head] + [_row(j, cfg) for j in sort_jobs(group)] + [""]
    if review:
        lines += ["### Not screened yet",
                  "These passed the keyword filters but weren't screened (AI off or the monthly budget is used up).", "",
                  "| Company | Role | Location | Found via |", "|---|---|---|---|"]
        lines += [f"| {_md(r['company'])} | [{_md(r['title'])}]({r['url']}) | {_md(r.get('location', ''))} | {_md(r.get('source', ''))} |"
                  for r in review[:60]] + [""]
    if closed_recent:
        lines += ["### Closed recently", ""] + [f"- {_md(c['name'])} ({c['date']})" for c in closed_recent[:15]] + [""]
    lines += [f"<details><summary>Screened out ({len(screened)})</summary>", ""]
    lines += [f"- [{_md(s['name'])}]({s['url']}): {_md(s['reason'])}" for s in screened[:200]]
    lines += ["", "</details>", "", END]
    return "\n".join(lines)


def write_readme(block: str) -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8") if path.exists() else f"# Offer board\n\n{START}\n{END}\n"
    if START in text and END in text:
        pre, rest = text.split(START, 1)
        post = rest.split(END, 1)[1]
        text = pre + block + post
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def write_html(jobs, screened, cfg, refreshed: str) -> None:
    tpl = (ROOT / "templates" / "board.html").read_text(encoding="utf-8")
    d = dt.date.fromisoformat(refreshed)
    meta = {
        "refreshed": refreshed, "refreshed_label": f"{d.strftime('%b')} {d.day}, {d.year}",
        "new_since": _new_since(), "offer_k": cfg["offer_line_k"], "candidate": cfg.get("candidate_profile", ""),
    }
    page_jobs = []
    for j in jobs:
        keep = {k: j.get(k) for k in ("id", "company", "track", "role", "location", "base", "est", "fit", "why",
                                      "caveat", "posted", "source", "url", "visa", "added", "stretch", "stale")}
        page_jobs.append({k: v for k, v in keep.items() if v not in (None, "")})
    payload = json.dumps({"meta": meta, "jobs": page_jobs, "screened": screened[:300]}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # keep the JSON from closing the script tag
    out = tpl.replace("/*__BOARD_DATA__*/null", payload, 1)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(out, encoding="utf-8")
