"""Run one board update.

    python -m scripts.update              # full run (needs ANTHROPIC_API_KEY for the AI steps)
    python -m scripts.update --no-ai      # free sources + link checks only
    python -m scripts.update --no-collect # only re-check links and re-render

Steps: re-check every role on the board, collect postings from all sources,
drop anything already seen, apply keyword rules, screen the rest with Claude,
then write data/*.json, README.md and docs/index.html and open a GitHub issue
when roles are added or removed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re

from . import notify, render, sources, verify
from .ai import AI, ai_available
from .ingest import ingest
from .rules import _PHD, _num, decide, ident, same_role, tkey
from .common import DATA, days_between, job_key, load_config, load_json, prefilter, save_json, today

SOURCE_PRIORITY = ("Greenhouse", "Lever", "Ashby", "Workday", "Google", "Claude", "speedyapply", "Simplify")
INTERESTING_REASONS = ("below your line", "more experience", "sponsor", "citizenship")


class Log:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        print(msg, flush=True)
        self.lines.append(msg)

    def warn(self, msg: str) -> None:
        print(f"WARN: {msg}", flush=True)
        self.lines.append(f"WARN: {msg}")


def priority(c: dict) -> int:
    src = c.get("source", "")
    for i, name in enumerate(SOURCE_PRIORITY):
        if src.startswith(name):
            return i
    return len(SOURCE_PRIORITY)


def clean_title(t: str) -> str:
    return re.split(r"\s+[—–|-]\s+(?:Google Careers|Careers|Jobs)\b", t or "")[0].strip()


def slim(c: dict) -> dict:
    return {k: c.get(k, "") for k in ("company", "title", "url", "location", "source")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-ai", action="store_true", help="skip Claude discovery and screening")
    ap.add_argument("--no-collect", action="store_true", help="only re-check links and re-render")
    ap.add_argument("--max-screen", type=int, default=None)
    args = ap.parse_args(argv)

    cfg, log, run_date = load_config(), Log(), today()
    jobs = load_json(DATA / "jobs.json", [])
    screened = load_json(DATA / "screened.json", [])
    seen = load_json(DATA / "seen.json", {})
    state = load_json(DATA / "state.json", {})
    state["runs"] = int(state.get("runs", 0)) + 1

    # 1) Re-check every role on the board.
    kept, closed_now = [], []
    for j in jobs:
        verdict = verify.check_job(j)
        j["last_checked"] = run_date
        if verdict == "open":
            j["last_open"], j["check_failures"] = run_date, 0
        elif verdict == "closed":
            j["check_failures"] = int(j.get("check_failures", 0)) + 1
        j["stale"] = days_between(j.get("last_open", run_date), run_date) > cfg["verify"]["stale_after_days"]
        if j.get("check_failures", 0) >= cfg["verify"]["close_after_failures"]:
            entry = {"name": f"{j['company']}, {j['role']}", "url": j["url"], "date": run_date,
                     "reason": f"Closed: the posting stopped showing as open (checked {run_date})."}
            screened.insert(0, entry)
            closed_now.append(entry)
            seen[job_key(j["company"], j["role"], j["url"])] = {"date": run_date, "decision": "closed",
                                                                "tkey": tkey(j["company"], j["role"])}
        else:
            kept.append(j)
        log.info(f"check {verdict:7s} {j['company']}: {j['role'][:60]}")
    jobs = kept

    # 1b) Merge roles that Claude Code sweeps or routines left in data/inbox/.
    ingested = ingest(cfg, jobs, screened, seen, run_date, log)

    # 2) Collect from every source, plus Claude discovery.
    candidates = [] if args.no_collect else sources.collect(cfg, log)
    ai = None
    if not args.no_ai and not args.no_collect and ai_available(cfg):
        try:
            ai = AI(cfg, state, log)
        except Exception as e:  # noqa: BLE001
            log.warn(f"AI disabled for this run: {e!r}")
    if ai and ai.can_spend(0.5):
        queries = cfg["sources"].get("discovery_queries", []) or []
        if queries:
            n = min(int(cfg["ai"].get("queries_per_run", 10)), len(queries))
            cursor = int(state.get("discovery_cursor", 0)) % len(queries)
            batch = [queries[(cursor + i) % len(queries)] for i in range(n)]
            state["discovery_cursor"] = (cursor + n) % len(queries)
            for q in batch:
                found = ai.discover(q)
                log.info(f"discovery '{q[:55]}': {len(found)} postings")
                candidates.extend(found)
    elif ai:
        log.warn("monthly AI budget reached; skipping discovery")

    # 3) Drop what's already on the board or seen before, then apply keyword rules.
    board_keys = {job_key(j["company"], j["role"], j["url"]) for j in jobs}
    board_ids = {i for i in (ident(j) for j in jobs) if i}
    board_t = {tkey(j["company"], j["role"]) for j in jobs}
    seen_t = {v.get("tkey") for v in seen.values() if isinstance(v, dict)}
    # Employers often post a PhD and a non-PhD variant of the same role; keep the PhD one.
    def variant_key(c):
        return tkey(c.get("company", ""), _PHD.sub("", c.get("title", "")))
    phd_variants = {variant_key(c) for c in candidates if _PHD.search(c.get("title", ""))}
    candidates = [c for c in candidates if _PHD.search(c.get("title", "")) or variant_key(c) not in phd_variants]
    fresh, batch_keys, rule_drops = [], set(), 0
    for c in sorted(candidates, key=priority):
        if not c.get("url") or not c.get("title"):
            continue
        k, t = job_key(c.get("company", ""), c["title"], c["url"]), tkey(c.get("company", ""), c["title"])
        cid = ident(c)
        if (k in batch_keys or t in batch_keys or k in board_keys or t in board_t or k in seen or t in seen_t
                or (cid and (cid in board_ids or cid in batch_keys))
                or any(same_role(c.get("company", ""), c["title"], j["company"], j["role"]) for j in jobs)):
            continue
        batch_keys.update({k, t} | ({cid} if cid else set()))
        ok, _why = prefilter(c, cfg)
        if ok:
            fresh.append((k, t, c))
        else:
            rule_drops += 1
    log.info(f"{len(candidates)} postings collected, {rule_drops} dropped by rules, {len(fresh)} to screen")

    # 4) Screen with Claude (bounded per run).
    max_screen = args.max_screen if args.max_screen is not None else int(cfg["ai"].get("max_screen_per_run", 60))
    new_jobs, unscreened = list(ingested), []
    for i, (k, t, c) in enumerate(fresh):
        if ai is None or i >= max_screen or not ai.can_spend(0.05):
            unscreened.append(slim(c))
            continue
        text, page_title = sources.fetch_posting_text(c)
        if c.get("needs_detail") and page_title:
            c["title"] = clean_title(page_title)
        triage_model = cfg["ai"].get("triage_model")
        final_model = cfg["ai"]["screen_model"]
        r = ai.screen(c, text, model=triage_model or final_model)
        if r is None:
            unscreened.append(slim(c))
            continue
        decision, reason, job = decide(r, c, cfg, run_date)
        if decision == "added" and triage_model and triage_model != final_model and ai.can_spend(0.05):
            # Second opinion from the stronger model for anything that passes triage.
            r2 = ai.screen(c, text, model=final_model)
            if r2 is not None:
                r = r2
                decision, reason, job = decide(r, c, cfg, run_date)
        seen[k] = {"date": run_date, "decision": decision, "reason": reason, "tkey": t}
        if decision == "added":
            if any(x["id"] == job["id"] for x in jobs):
                job["id"] += f"-{i}"
            jobs.append(job)
            new_jobs.append(job)
            log.info(f"ADDED {job['company']}: {job['role']} (${job['est'][0]}K–${job['est'][1]}K, fit {job['fit']})")
        elif int(_num(r.get("fit"), 0)) >= 3 and any(w in reason for w in INTERESTING_REASONS):
            screened.insert(0, {"name": f"{r.get('company') or c['company']}, {r.get('role') or c['title']}",
                                "reason": reason[0].upper() + reason[1:] + ".", "url": c["url"], "date": run_date})
    state["review"] = unscreened[:100]

    # 5) Housekeeping, render, notify, save.
    cutoff = (dt.date.fromisoformat(run_date) - dt.timedelta(days=150)).isoformat()
    seen = {k: v for k, v in seen.items() if isinstance(v, dict) and v.get("date", "") >= cutoff}
    screened = screened[:300]
    state["last_run"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if ai:
        log.info(f"AI cost this run ${ai.run_cost:.3f} ({ai.searches} searches); month to date ${ai.month_spend:.2f}")
    closed_recent = [s for s in screened if s.get("reason", "").startswith("Closed:")
                     and days_between(s.get("date", run_date), run_date) <= 14]
    render.write_readme(render.readme_block(jobs, screened, state.get("review", []), closed_recent, state, cfg))
    render.write_html(jobs, screened, cfg, run_date)
    notify.post_issue(new_jobs, closed_now, cfg, run_date, log)
    save_json(DATA / "jobs.json", jobs)
    save_json(DATA / "screened.json", screened)
    save_json(DATA / "seen.json", seen)
    save_json(DATA / "state.json", state)
    log.info(f"done: {len(jobs)} roles on the board, {len(new_jobs)} new, {len(closed_now)} closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
