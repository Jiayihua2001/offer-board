"""Merge roles found by hand or by Claude Code (sweeps, routines) into the board.

Write findings to data/inbox/<YYYY-MM-DD>-<label>.json, then run

    python -m scripts.ingest

Every role passes the same rules as automated screening (US only, visa terms,
no quant, offer line, level) and the same duplicate checks. Processed files move
to data/inbox/processed/. The GitHub workflow also ingests the inbox at the start
of every run, so committing an inbox file is enough when the workflow is active.

File format:
{
  "source": "Claude Code sweep",
  "add": [ { "company": "...", "role": "...", "url": "https://...", "location": "...",
             "posted_pay": "...", "est_low_k": 280, "est_high_k": 360, "fit": 4,
             "track": "science" | "llm", "level": "phd_new_grad" | "early_career" | "senior",
             "in_us": true, "open": true, "quant": false, "restricted": false,
             "sponsorship": "yes" | "no" | "unclear", "posted": "YYYY-MM-DD",
             "why": "...", "caveat": "..." } ],
  "screened": [ { "company": "...", "role": "...", "url": "...", "reason": "..." } ]
}
"""
from __future__ import annotations

import shutil

from . import render
from .common import DATA, job_key, load_config, load_json, save_json, today
from .rules import decide, ident, same_role, tkey

INBOX = DATA / "inbox"


def _is_duplicate(job: dict, jobs: list[dict], seen: dict) -> bool:
    k, t, i = job_key(job["company"], job["role"], job["url"]), tkey(job["company"], job["role"]), ident(job)
    seen_t = {v.get("tkey") for v in seen.values() if isinstance(v, dict) and v.get("decision") == "closed"}
    for j in jobs:
        if (k == job_key(j["company"], j["role"], j["url"]) or t == tkey(j["company"], j["role"])
                or (i and i == ident(j)) or same_role(job["company"], job["role"], j["company"], j["role"])):
            return True
    return (seen.get(k, {}).get("decision") == "closed") or t in seen_t


def ingest(cfg: dict, jobs: list[dict], screened: list[dict], seen: dict, run_date: str, log) -> list[dict]:
    """Merge every inbox file. Mutates jobs/screened/seen and returns the roles added."""
    if not INBOX.exists():
        return []
    added = []
    for path in sorted(INBOX.glob("*.json")):
        payload = load_json(path, None)
        if payload is None:
            log.warn(f"inbox file {path.name} is not valid JSON; left in place")
            continue
        if isinstance(payload, list):
            payload = {"add": payload}
        source = str(payload.get("source") or f"inbox: {path.name}")[:80]
        n_add = n_skip = 0
        for item in payload.get("add") or []:
            if not isinstance(item, dict) or not str(item.get("url", "")).startswith("http"):
                n_skip += 1
                continue
            c = {"url": item["url"], "company": item.get("company", ""), "title": item.get("role", ""),
                 "location": item.get("location", ""), "source": source, "pay_hint": "",
                 "posted": str(item.get("posted", "")), "ats": None}
            decision, reason, job = decide(item, c, cfg, run_date)
            key = job_key(c["company"], c["title"], c["url"])
            if decision != "added":
                screened.insert(0, {"name": f"{c['company']}, {c['title']}", "url": c["url"], "date": run_date,
                                    "reason": reason[0].upper() + reason[1:] + "."})
                seen[key] = {"date": run_date, "decision": "screened", "reason": reason, "tkey": tkey(c["company"], c["title"])}
                n_skip += 1
                continue
            if _is_duplicate(job, jobs, seen):
                n_skip += 1
                continue
            if any(x["id"] == job["id"] for x in jobs):
                job["id"] += f"-{len(jobs)}"
            jobs.append(job)
            added.append(job)
            seen[key] = {"date": run_date, "decision": "added", "reason": "", "tkey": tkey(job["company"], job["role"])}
            n_add += 1
        for s in payload.get("screened") or []:
            if not isinstance(s, dict) or not s.get("url"):
                continue
            company, role = s.get("company", ""), s.get("role", "")
            reason = str(s.get("reason") or "Screened out").strip().rstrip(".")
            screened.insert(0, {"name": f"{company}, {role}".strip(", "), "url": s["url"], "date": run_date,
                                "reason": reason[0].upper() + reason[1:] + "."})
            seen[job_key(company, role, s["url"])] = {"date": run_date, "decision": "screened", "reason": reason,
                                                      "tkey": tkey(company, role)}
        (INBOX / "processed").mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(INBOX / "processed" / path.name))
        log.info(f"inbox {path.name}: {n_add} added, {n_skip} skipped")
    return added


def main() -> int:
    from .update import Log  # local import keeps module load order simple

    cfg, log, run_date = load_config(), Log(), today()
    jobs = load_json(DATA / "jobs.json", [])
    screened = load_json(DATA / "screened.json", [])
    seen = load_json(DATA / "seen.json", {})
    state = load_json(DATA / "state.json", {})
    added = ingest(cfg, jobs, screened, seen, run_date, log)
    reviewed = {s["url"] for s in screened} | {j["url"] for j in jobs}
    state["review"] = [r for r in state.get("review", []) if r.get("url") not in reviewed]
    closed_recent = [s for s in screened if s.get("reason", "").startswith("Closed:")][:15]
    render.write_readme(render.readme_block(jobs, screened[:300], state.get("review", []), closed_recent, state, cfg))
    render.write_html(jobs, screened[:300], cfg, run_date)
    save_json(DATA / "jobs.json", jobs)
    save_json(DATA / "screened.json", screened[:300])
    save_json(DATA / "seen.json", seen)
    save_json(DATA / "state.json", state)
    for j in added:
        log.info(f"ADDED {j['company']}: {j['role']} (${j['est'][0]}K–${j['est'][1]}K, fit {j['fit']})")
    log.info(f"ingest done: {len(added)} added, {len(jobs)} roles on the board")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
