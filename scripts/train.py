"""Interview training: daily assignments, a test runner, a quiz, progress tracking and the training page.

    python -m scripts.train today               # today's assignment and quiz cards (recorded, so it's stable all day)
    python -m scripts.train start <id>          # copy the starter to training/work/<id>.py and print the problem
    python -m scripts.train test <id>           # run the assignment's tests against training/work/<id>.py
    python -m scripts.train done <id> [--minutes N] [--score 85] [--mock] [--note "..."]
    python -m scripts.train skip <id> --reason "..."
    python -m scripts.train hint <id> [n]       # show hint n (default 1)
    python -m scripts.train quiz [--n 6]        # today's flashcards, with answers, for a Claude-run session
    python -m scripts.train quiz --grade q-id=good q-id=again ...
    python -m scripts.train mock [--track leetcode] [--minutes 45]   # timed practice on a random open problem
    python -m scripts.train status | plan | log
    python -m scripts.train goal --title "..." --target 2027-01-31
    python -m scripts.train selfcheck [ids...]  # run every test file against the reference solutions
    python -m scripts.train render | open       # rebuild docs/train.html and open it
    python -m scripts.train sync <file|base64|-> # merge events exported from the page
    python -m scripts.train new <id> --track leetcode --module lc-arrays --title "..."  # scaffold an assignment

All state lives in training/progress.json as an append-only list of events; everything
else (status per assignment, quiz boxes, streaks) is derived from it, both here and in the page.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import os
import random
import re
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from .common import ROOT, load_json, save_json

TRAINING = ROOT / "training"
ASSIGN = TRAINING / "assignments"
WORK = TRAINING / "work"
PROGRESS = TRAINING / "progress.json"
CURRICULUM = TRAINING / "curriculum.json"
QUIZ = TRAINING / "quiz.json"
PAGE = ROOT / "docs" / "train.html"
TEMPLATE = ROOT / "templates" / "train.html"

LEITNER_DAYS = [1, 2, 4, 8, 16, 32]          # days until a card comes back, by box
GRADE_STEP = {"again": None, "hard": 0, "good": 1, "easy": 2}
DEFAULT_SETTINGS = {
    "goal": "Ready for research scientist and applied scientist interview loops",
    "start_date": "2026-10-26",
    "target_date": "2027-01-31",
    "rotation": ["leetcode", "ml", "transformer", "leetcode", "ml", "transformer", "review"],
    "quiz_per_day": 6,
    "mock_minutes": 45,
}
STATUS_WORD = {"todo": "to do", "in_progress": "in progress", "done": "done", "skipped": "skipped"}


# ---------------------------------------------------------------------- runner
# The same code runs locally (in a subprocess) and in the page (in Pyodide), so the two agree.
RUNNER_SRC = r'''
import io, json, sys, time, traceback, types

def _load_module(name, src, filename):
    mod = types.ModuleType(name)
    mod.__file__ = filename
    sys.modules[name] = mod
    exec(compile(src, filename, "exec"), mod.__dict__)
    return mod

def _format_error(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename in ("solution.py", "tests.py")]
    lines = []
    for f in frames[-4:]:
        lines.append("  %s line %d, in %s" % (f.filename, f.lineno, f.name))
        if f.line:
            lines.append("    " + f.line.strip())
    msg = str(exc).strip()
    head = type(exc).__name__ + (": " + msg if msg else "")
    return "\n".join(lines + [head]) if lines else head

def run_tests(solution_src, tests_src):
    out = {"ok": False, "passed": 0, "total": 0, "results": [], "error": "", "stdout": ""}
    for name in ("solution", "tests_module"):
        sys.modules.pop(name, None)
    real_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        _load_module("solution", solution_src, "solution.py")
        tests = _load_module("tests_module", tests_src, "tests.py")
    except BaseException as exc:
        sys.stdout = real_stdout
        out["error"] = "Could not import your solution or the tests:\n" + _format_error(exc)
        out["stdout"] = buf.getvalue()[-3000:]
        return out
    sys.stdout = real_stdout
    out["stdout"] = buf.getvalue()[-3000:]
    names = [n for n in dir(tests) if n.startswith("test_") and callable(getattr(tests, n))]
    names.sort(key=lambda n: getattr(getattr(tests, n), "__code__", None) and getattr(tests, n).__code__.co_firstlineno or 0)
    for n in names:
        fn = getattr(tests, n)
        rec = {"name": n, "ok": False, "ms": 0, "error": "", "stdout": ""}
        buf = io.StringIO()
        sys.stdout = buf
        t0 = time.perf_counter()
        try:
            fn()
            rec["ok"] = True
        except BaseException as exc:
            rec["error"] = _format_error(exc)
        finally:
            sys.stdout = real_stdout
        rec["ms"] = int((time.perf_counter() - t0) * 1000)
        rec["stdout"] = buf.getvalue()[-2000:]
        out["results"].append(rec)
    out["total"] = len(names)
    out["passed"] = sum(1 for r in out["results"] if r["ok"])
    out["ok"] = out["total"] > 0 and out["passed"] == out["total"]
    return out
'''

_RUNNER_MAIN = RUNNER_SRC + r'''
if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    print("__RESULT__" + json.dumps(run_tests(payload["solution"], payload["tests"])))
'''


def run_tests_local(solution_src: str, tests_src: str, timeout: int = 60) -> dict:
    try:
        proc = subprocess.run([sys.executable, "-c", _RUNNER_MAIN], input=json.dumps({"solution": solution_src, "tests": tests_src}),
                              capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"ok": False, "passed": 0, "total": 0, "results": [],
                "error": f"Timed out after {timeout}s. An infinite loop, or something far too slow?", "stdout": ""}
    marker = proc.stdout.rfind("__RESULT__")
    if marker < 0:
        return {"ok": False, "passed": 0, "total": 0, "results": [],
                "error": "The test process crashed:\n" + (proc.stderr or proc.stdout)[-3000:], "stdout": ""}
    out = json.loads(proc.stdout[marker + len("__RESULT__"):])
    extra = proc.stdout[:marker].strip()
    if extra:
        out["stdout"] = (extra + "\n" + out.get("stdout", ""))[-3000:]
    if proc.stderr.strip() and not out.get("ok"):
        out["stderr"] = proc.stderr[-2000:]
    return out


def format_report(res: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    if res.get("error"):
        lines.append(res["error"])
    for r in res.get("results", []):
        lines.append(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['name']} ({r['ms']} ms)")
        if not r["ok"] and r.get("error"):
            lines.extend("        " + ln for ln in r["error"].splitlines())
        if r.get("stdout") and not r["ok"]:
            lines.extend("        > " + ln for ln in r["stdout"].strip().splitlines()[-8:])
    if res.get("stdout", "").strip():
        lines.extend("  > " + ln for ln in res["stdout"].strip().splitlines()[-8:])
    if res.get("stderr"):
        lines.extend("  ! " + ln for ln in res["stderr"].strip().splitlines()[-8:])
    if res.get("total"):
        lines.append(f"  {res['passed']} of {res['total']} tests passed" + ("  ✓" if res["ok"] else ""))
    return "\n".join(lines)


# -------------------------------------------------------------------- loading
def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def today_str() -> str:
    return dt.date.today().isoformat()


def new_id() -> str:
    return f"{int(time.time() * 1000):x}-{random.randrange(16 ** 4):04x}"


def load_curriculum() -> dict:
    cur = load_json(CURRICULUM, None)
    if not cur:
        sys.exit(f"missing {CURRICULUM}")
    return cur


def load_quiz() -> list[dict]:
    data = load_json(QUIZ, {})
    return data.get("cards", []) if isinstance(data, dict) else data


def load_assignments() -> dict[str, dict]:
    out = {}
    if not ASSIGN.exists():
        return out
    for d in sorted(ASSIGN.iterdir()):
        meta_path = d / "meta.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        meta = load_json(meta_path, None)
        if not isinstance(meta, dict):
            print(f"WARN: {meta_path} is not valid JSON", file=sys.stderr)
            continue
        meta.setdefault("id", d.name)
        meta["dir"] = str(d)
        meta.setdefault("kind", "code")
        meta.setdefault("runtime", "python")
        meta.setdefault("difficulty", "medium")
        meta.setdefault("minutes", 30)
        meta.setdefault("hints", [])
        meta.setdefault("tags", [])
        meta.setdefault("order", 999)
        out[meta["id"]] = meta
    return out


def read_file(meta: dict, name: str) -> str:
    p = Path(meta["dir"]) / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def ordered_ids(cur: dict, assignments: dict) -> dict[str, list[str]]:
    """Assignment ids per track, in module order then meta.order then id."""
    by_track = {}
    for tr in cur.get("tracks", []):
        mods = {m["id"]: i for i, m in enumerate(tr.get("modules", []))}
        ids = [a["id"] for a in assignments.values() if a.get("track") == tr["id"]]
        ids.sort(key=lambda i: (mods.get(assignments[i].get("module"), 999), assignments[i].get("order", 999), i))
        by_track[tr["id"]] = ids
    return by_track


def load_progress() -> dict:
    p = load_json(PROGRESS, None)
    if not isinstance(p, dict):
        p = {"version": 1, "settings": dict(DEFAULT_SETTINGS), "events": []}
    p.setdefault("settings", dict(DEFAULT_SETTINGS))
    for k, v in DEFAULT_SETTINGS.items():
        p["settings"].setdefault(k, v)
    p.setdefault("events", [])
    return p


def save_progress(p: dict) -> None:
    p["events"].sort(key=lambda e: (e.get("ts", ""), e.get("id", "")))
    save_json(PROGRESS, p)


def add_event(p: dict, **fields) -> dict:
    e = {"id": new_id(), "ts": now_iso(), "date": today_str()}
    e.update(fields)
    p["events"].append(e)
    return e


# -------------------------------------------------------------------- derive
def reduce_events(events: list[dict], settings: dict) -> dict:
    st = {"assignments": {}, "quiz": {}, "days": {}, "settings": dict(settings), "by_date": {}}
    for e in sorted(events, key=lambda e: (e.get("ts", ""), e.get("id", ""))):
        t = e.get("type")
        date = e.get("date", "")
        if t == "settings":
            st["settings"].update(e.get("settings") or {})
        elif t == "plan":
            st["days"][e.get("date", "")] = {"track": e.get("track"), "aid": e.get("aid"), "quiz": list(e.get("quiz") or []),
                                             "mode": e.get("mode", "practice")}
        elif t in ("start", "test", "done", "skip"):
            a = st["assignments"].setdefault(e.get("aid"), {"status": "todo", "attempts": 0, "best": 0.0, "minutes": 0,
                                                              "last": "", "mock": False, "started": "", "mode": "practice",
                                                              "deadline": "", "done_count": 0, "reason": "", "note": ""})
            if t == "start":
                if a["status"] != "done":
                    a["status"] = "in_progress"
                a["started"], a["mode"], a["deadline"] = e.get("ts", ""), e.get("mode", "practice"), e.get("deadline", "")
            elif t == "test":
                a["attempts"] += 1
                a["last_test"] = [e.get("passed", 0), e.get("total", 0)]
            elif t == "done":
                a["status"] = "done"
                a["best"] = max(a["best"], float(e.get("score") or 0))
                a["minutes"] += int(e.get("minutes") or 0)
                a["last"] = date
                a["mock"] = a["mock"] or e.get("mode") == "mock"
                a["done_count"] += 1
                a["started"] = ""
                if e.get("note"):
                    a["note"] = e["note"]
                day = st["by_date"].setdefault(date, {"done": 0, "quiz": 0, "minutes": 0})
                day["done"] += 1
                day["minutes"] += int(e.get("minutes") or 0)
            elif t == "skip":
                if a["status"] != "done":
                    a["status"] = "skipped"
                    a["reason"] = e.get("reason", "")
        elif t == "quiz":
            q = st["quiz"].setdefault(e.get("qid"), {"box": 0, "due": "", "seen": 0, "last": ""})
            q["box"], q["due"], q["last"] = int(e.get("box") or 0), e.get("due", ""), e.get("grade", "")
            q["seen"] += 1
            day = st["by_date"].setdefault(date, {"done": 0, "quiz": 0, "minutes": 0})
            day["quiz"] += 1
    return st


def streak(st: dict, today: str) -> int:
    active = {d for d, v in st["by_date"].items() if v["done"] or v["quiz"]}
    d = dt.date.fromisoformat(today)
    if today not in active:
        d -= dt.timedelta(days=1)
    n = 0
    while d.isoformat() in active:
        n += 1
        d -= dt.timedelta(days=1)
    return n


def status_of(st: dict, aid: str) -> str:
    return st["assignments"].get(aid, {}).get("status", "todo")


def next_in_track(track: str, order: dict, st: dict, exclude=()) -> str | None:
    ids = [i for i in order.get(track, []) if i not in exclude]
    for want in ("todo", "in_progress", "skipped"):
        for i in ids:
            if status_of(st, i) == want:
                return i
    return None


def quiz_cards_for(date: str, st: dict, cards: list[dict], n: int) -> list[str]:
    due = [c["id"] for c in cards if c["id"] in st["quiz"] and (st["quiz"][c["id"]].get("due") or "") <= date]
    due.sort(key=lambda q: st["quiz"][q].get("due", ""))
    fresh = [c["id"] for c in cards if c["id"] not in st["quiz"]]
    return (due + fresh)[:n]


def plan_for(date: str, st: dict, cur: dict, assignments: dict, cards: list[dict], simulate_done=()) -> dict:
    """Deterministic plan for a date. Does not record anything."""
    if date in st["days"] and not simulate_done:
        return dict(st["days"][date])
    settings = st["settings"]
    rotation = settings.get("rotation") or DEFAULT_SETTINGS["rotation"]
    weekday = dt.date.fromisoformat(date).weekday()
    track = rotation[weekday % len(rotation)]
    order = ordered_ids(cur, assignments)
    tracks = [t["id"] for t in cur.get("tracks", [])]
    aid, mode = None, "practice"
    if track == "review":
        weak = [i for i, a in st["assignments"].items() if a["status"] == "done" and a["best"] < 1 and i in assignments]
        done = [i for i, a in st["assignments"].items() if a["status"] == "done" and i in assignments and i not in simulate_done]
        if weak:
            aid, mode = sorted(weak, key=lambda i: st["assignments"][i]["last"])[0], "review"
        elif done:
            rng = random.Random(date)
            aid, mode = rng.choice(sorted(done)), "mock"
    if aid is None:
        seq = [track] if track != "review" else []
        seq += [t for t in tracks if t not in seq]
        for t in seq:
            aid = next_in_track(t, order, st, exclude=simulate_done)
            if aid:
                track = t
                break
    return {"track": track, "aid": aid, "quiz": quiz_cards_for(date, st, cards, int(settings.get("quiz_per_day") or 6)), "mode": mode}


# --------------------------------------------------------------------- output
def days_left(st: dict, today: str) -> int:
    try:
        return (dt.date.fromisoformat(st["settings"]["target_date"]) - dt.date.fromisoformat(today)).days
    except (KeyError, ValueError):
        return 0


def day_number(st: dict, today: str) -> int:
    try:
        return (dt.date.fromisoformat(today) - dt.date.fromisoformat(st["settings"]["start_date"])).days + 1
    except (KeyError, ValueError):
        return 1


def describe(a: dict) -> str:
    return f"[{a.get('track')} · {a.get('difficulty')} · ~{a.get('minutes')} min · {a.get('runtime')}]"


def ctx():
    cur, assignments, cards, p = load_curriculum(), load_assignments(), load_quiz(), load_progress()
    st = reduce_events(p["events"], p["settings"])
    return cur, assignments, cards, p, st


def need(assignments: dict, aid: str) -> dict:
    if aid not in assignments:
        close = [i for i in assignments if aid.lower() in i]
        hint = f" Did you mean: {', '.join(close[:5])}?" if close else ""
        sys.exit(f"No assignment named {aid!r}.{hint}")
    return assignments[aid]


def work_path(aid: str) -> Path:
    return WORK / f"{aid}.py"


def cmd_today(args):
    cur, assignments, cards, p, st = ctx()
    today = today_str()
    plan = plan_for(today, st, cur, assignments, cards)
    if today not in st["days"]:
        add_event(p, type="plan", track=plan["track"], aid=plan["aid"], quiz=plan["quiz"], mode=plan.get("mode", "practice"))
        save_progress(p)
        st = reduce_events(p["events"], p["settings"])
        render_page(quiet=True)
    if args.json:
        a = assignments.get(plan["aid"]) if plan["aid"] else None
        print(json.dumps({"date": today, "plan": plan, "assignment": {k: a[k] for k in a if k != "dir"} if a else None,
                          "status": st["assignments"].get(plan["aid"], {}), "cards": [c for c in cards if c["id"] in plan["quiz"]],
                          "streak": streak(st, today), "days_left": days_left(st, today)}, indent=2))
        return
    d = dt.date.fromisoformat(today)
    print(f"Today, {d.strftime('%a %b %-d')}  ·  day {day_number(st, today)} of the plan  ·  streak {streak(st, today)}  ·  "
          f"{days_left(st, today)} days to {st['settings']['target_date']}")
    print(f"Goal: {st['settings']['goal']}")
    print()
    aid = plan["aid"]
    if aid and aid in assignments:
        a = assignments[aid]
        s = st["assignments"].get(aid, {})
        word = STATUS_WORD.get(s.get("status", "todo"), "to do")
        mode = {"review": "Review (you scored below 100% last time)", "mock": "Timed redo of a solved problem"}.get(plan.get("mode"), "Assignment")
        print(f"{mode}  {describe(a)}")
        print(f"  {aid}: {a.get('title')}   ({word}{', best ' + str(round(s['best'] * 100)) + '%' if s.get('best') else ''})")
        print(f"  start   python -m scripts.train start {aid}")
        if a.get("kind") == "code":
            print(f"  test    python -m scripts.train test {aid}")
        print(f"  done    python -m scripts.train done {aid}" + ("" if a.get("kind") == "code" else " --score 80"))
    else:
        print("No assignment left in the rotation. Add some with `python -m scripts.train new`.")
    q = plan["quiz"]
    n_due = sum(1 for i in q if i in st["quiz"])
    print()
    print(f"Quiz: {len(q)} cards ({len(q) - n_due} new, {n_due} due)   python -m scripts.train quiz")
    print(f"Page: {PAGE}")


def cmd_start(args):
    cur, assignments, cards, p, st = ctx()
    a = need(assignments, args.id)
    WORK.mkdir(parents=True, exist_ok=True)
    wp = work_path(a["id"])
    if a.get("kind") == "code":
        if not wp.exists() or args.force:
            wp.write_text(read_file(a, "starter.py"), encoding="utf-8")
            print(f"Starter copied to {wp}")
        else:
            print(f"Keeping your existing work in {wp} (use --force to reset it)")
    deadline = ""
    if args.minutes:
        deadline = (dt.datetime.now().astimezone() + dt.timedelta(minutes=args.minutes)).isoformat(timespec="seconds")
    add_event(p, type="start", aid=a["id"], mode="mock" if args.minutes else "practice", deadline=deadline)
    save_progress(p)
    render_page(quiet=True)
    print()
    print(f"# {a.get('title')}  {describe(a)}")
    print()
    print(readme_body(a))
    if a.get("kind") != "code" and a.get("rubric"):
        print("\nRubric (score yourself, or have Claude score you):")
        for i, r in enumerate(a["rubric"], 1):
            print(f"  {i}. {r}")
    if deadline:
        print(f"\nTimer: {args.minutes} minutes, until {deadline[11:16]}. Finish with: python -m scripts.train done {a['id']} --mock")
    if a.get("hints"):
        print(f"\n{len(a['hints'])} hints available: python -m scripts.train hint {a['id']} [n]")


def readme_body(a: dict) -> str:
    """The README without its first H1 line (the title is printed separately)."""
    text = read_file(a, "README.md").strip()
    return re.sub(r"^#\s+[^\n]*\n+", "", text, count=1)


def cmd_show(args):
    cur, assignments, cards, p, st = ctx()
    a = need(assignments, args.id)
    print(f"# {a.get('title')}  {describe(a)}\n")
    print(readme_body(a))
    if args.tests:
        print("\n--- tests.py ---\n" + read_file(a, "tests.py"))
    if args.solution:
        print("\n--- reference solution ---\n" + read_file(a, "solution.py"))


def cmd_hint(args):
    cur, assignments, cards, p, st = ctx()
    a = need(assignments, args.id)
    hints = a.get("hints") or []
    if not hints:
        print("No hints for this one. Ask Claude for a nudge instead.")
        return
    n = max(1, min(args.n, len(hints)))
    for i in range(n):
        print(f"Hint {i + 1} of {len(hints)}: {hints[i]}")


def _run_for(a: dict, src: str) -> dict:
    return run_tests_local(src, read_file(a, "tests.py"), timeout=int(a.get("timeout") or 60))


def cmd_test(args):
    cur, assignments, cards, p, st = ctx()
    a = need(assignments, args.id)
    if a.get("kind") != "code":
        sys.exit("This is a written assignment; there are no tests. Record it with `done <id> --score N`.")
    wp = work_path(a["id"])
    if not wp.exists():
        sys.exit(f"No work file yet. Run: python -m scripts.train start {a['id']}")
    res = _run_for(a, wp.read_text(encoding="utf-8"))
    print(format_report(res, f"{a['id']}: {a.get('title')}"))
    add_event(p, type="test", aid=a["id"], passed=res.get("passed", 0), total=res.get("total", 0))
    save_progress(p)
    render_page(quiet=True)
    if args.json:
        print(json.dumps(res))
    sys.exit(0 if res.get("ok") else 1)


def cmd_done(args):
    cur, assignments, cards, p, st = ctx()
    a = need(assignments, args.id)
    s = st["assignments"].get(a["id"], {})
    score = None
    if a.get("kind") == "code":
        wp = work_path(a["id"])
        if not wp.exists():
            sys.exit(f"No work file yet. Run: python -m scripts.train start {a['id']}")
        res = _run_for(a, wp.read_text(encoding="utf-8"))
        print(format_report(res, f"{a['id']}: {a.get('title')}"))
        score = (res.get("passed", 0) / res["total"]) if res.get("total") else 0.0
        if not res.get("ok") and not args.force:
            sys.exit("Not all tests pass. Fix them, or record it anyway with --force.")
    if args.score is not None:
        score = max(0.0, min(1.0, args.score / 100.0 if args.score > 1 else args.score))
    if score is None:
        sys.exit("Written assignments need a score: --score 0-100")
    minutes = args.minutes
    mode = "mock" if args.mock or s.get("mode") == "mock" else "practice"
    if minutes is None and s.get("started") and s.get("status") == "in_progress":
        try:
            started = dt.datetime.fromisoformat(s["started"])
            minutes = int((dt.datetime.now().astimezone() - started).total_seconds() // 60)
            minutes = max(1, min(minutes, 300))
        except ValueError:
            minutes = None
    if minutes is None:
        minutes = int(a.get("minutes") or 30)
    add_event(p, type="done", aid=a["id"], score=round(score, 3), minutes=minutes, mode=mode, note=args.note or "", where="cli")
    save_progress(p)
    st = reduce_events(p["events"], p["settings"])
    render_page(quiet=True)
    print(f"Recorded {a['id']}: {round(score * 100)}% in {minutes} min ({mode}). Streak {streak(st, today_str())}. "
          f"{sum(1 for x in st['assignments'].values() if x['status'] == 'done')} of {len(assignments)} assignments done.")


def cmd_skip(args):
    cur, assignments, cards, p, st = ctx()
    a = need(assignments, args.id)
    add_event(p, type="skip", aid=a["id"], reason=args.reason or "")
    save_progress(p)
    render_page(quiet=True)
    print(f"Skipped {a['id']}. It comes back after the rest of its track.")


def _apply_grade(q: dict, grade: str, today: str) -> tuple[int, str]:
    box = int(q.get("box") or 0)
    step = GRADE_STEP.get(grade, 1)
    box = 0 if step is None else min(len(LEITNER_DAYS) - 1, box + step)
    due = (dt.date.fromisoformat(today) + dt.timedelta(days=LEITNER_DAYS[box])).isoformat()
    return box, due


def cmd_quiz(args):
    cur, assignments, cards, p, st = ctx()
    today = today_str()
    by_id = {c["id"]: c for c in cards}
    if args.grade:
        n = 0
        for item in args.grade:
            if "=" not in item:
                sys.exit(f"grades look like q-id=good, got {item!r}")
            qid, grade = item.split("=", 1)
            grade = grade.strip().lower()
            if qid not in by_id or grade not in GRADE_STEP:
                sys.exit(f"unknown card or grade: {item!r} (grades: again, hard, good, easy)")
            box, due = _apply_grade(st["quiz"].get(qid, {}), grade, today)
            add_event(p, type="quiz", qid=qid, grade=grade, box=box, due=due)
            st["quiz"][qid] = {"box": box, "due": due}
            n += 1
        save_progress(p)
        render_page(quiet=True)
        st = reduce_events(p["events"], p["settings"])
        print(f"Recorded {n} grades. Streak {streak(st, today)}. "
              f"{sum(1 for q in st['quiz'].values() if (q.get('due') or '') <= today)} cards due now.")
        return
    plan = plan_for(today, st, cur, assignments, cards)
    ids = plan["quiz"] if not args.all else [c["id"] for c in cards]
    if args.n:
        ids = quiz_cards_for(today, st, cards, args.n)
    if args.json:
        print(json.dumps([by_id[i] for i in ids if i in by_id], indent=2))
        return
    if not ids:
        print("No cards due today.")
        return
    print(f"{len(ids)} cards. Ask each question, let the answer come first, then grade: "
          f"python -m scripts.train quiz --grade {ids[0]}=good ...\n")
    for i, qid in enumerate(ids, 1):
        c = by_id.get(qid)
        if not c:
            continue
        q = st["quiz"].get(qid, {})
        state = f"box {q.get('box', 0)}, last {q.get('last')}" if q else "new"
        print(f"{i}. [{qid}] ({c.get('topic', '')}; {state})\n   Q: {c['q']}\n   A: {c['a']}\n")


def cmd_mock(args):
    cur, assignments, cards, p, st = ctx()
    order = ordered_ids(cur, assignments)
    pool_tracks = [args.track] if args.track else [t for t in order if t != "design"]
    pool = [i for t in pool_tracks for i in order.get(t, []) if assignments[i].get("kind") == "code"
            and (status_of(st, i) != "done" or args.any)]
    if not pool:
        sys.exit("Nothing left to mock in that track. Try --any to redo solved problems.")
    aid = random.choice(pool)
    minutes = args.minutes or int(st["settings"].get("mock_minutes") or 45)
    args.id, args.minutes, args.force = aid, minutes, False
    cmd_start(args)


def cmd_status(args):
    cur, assignments, cards, p, st = ctx()
    today = today_str()
    order = ordered_ids(cur, assignments)
    print(f"Goal: {st['settings']['goal']}")
    print(f"Target {st['settings']['target_date']} ({days_left(st, today)} days left) · day {day_number(st, today)} · "
          f"streak {streak(st, today)} · {sum(v['minutes'] for v in st['by_date'].values()) / 60:.1f} h logged")
    print()
    for tr in cur.get("tracks", []):
        ids = order.get(tr["id"], [])
        done = sum(1 for i in ids if status_of(st, i) == "done")
        prog = [i for i in ids if status_of(st, i) == "in_progress"]
        print(f"  {tr['title']:34s} {done:3d}/{len(ids):<3d}" + (f"  in progress: {', '.join(prog)}" if prog else ""))
    seen = len(st["quiz"])
    due = sum(1 for q in st["quiz"].values() if (q.get("due") or "") <= today)
    strong = sum(1 for q in st["quiz"].values() if int(q.get("box") or 0) >= 3)
    print(f"  {'Quiz cards':34s} {seen:3d}/{len(cards):<3d}  due {due}, solid {strong}")
    if args.json:
        print(json.dumps({"today": today, "streak": streak(st, today), "assignments": st["assignments"], "quiz": st["quiz"],
                          "settings": st["settings"]}, indent=2))


def cmd_plan(args):
    cur, assignments, cards, p, st = ctx()
    d = dt.date.today()
    done = []
    print(f"Next {args.days} days (assuming each day's assignment gets done):")
    for i in range(args.days):
        date = (d + dt.timedelta(days=i)).isoformat()
        plan = plan_for(date, st, cur, assignments, cards, simulate_done=tuple(done))
        aid = plan["aid"]
        if aid:
            done.append(aid)
        a = assignments.get(aid, {})
        print(f"  {date} {dt.date.fromisoformat(date).strftime('%a')}  {plan['track']:12s} {aid or '-':32s} {a.get('title', '')}")


def cmd_log(args):
    cur, assignments, cards, p, st = ctx()
    for e in p["events"][-args.n:]:
        extra = {k: v for k, v in e.items() if k not in ("id", "ts", "date", "type")}
        print(f"  {e['ts'][:16]}  {e['type']:8s} {json.dumps(extra, ensure_ascii=False)[:120]}")


def cmd_goal(args):
    cur, assignments, cards, p, st = ctx()
    s = {}
    if args.title:
        s["goal"] = args.title
    if args.target:
        dt.date.fromisoformat(args.target)
        s["target_date"] = args.target
    if args.rotation:
        parts = [x.strip() for x in args.rotation.split(",")]
        if len(parts) != 7:
            sys.exit("rotation needs 7 comma-separated entries, Monday first, e.g. leetcode,ml,transformer,leetcode,ml,transformer,review")
        s["rotation"] = parts
    if args.quiz_per_day:
        s["quiz_per_day"] = args.quiz_per_day
    if not s:
        print(json.dumps(st["settings"], indent=2))
        return
    add_event(p, type="settings", settings=s)
    p["settings"].update(s)
    save_progress(p)
    render_page(quiet=True)
    print("Updated: " + json.dumps(s))


def cmd_selfcheck(args):
    cur, assignments, cards, p, st = ctx()
    ids = args.ids or sorted(assignments)
    if args.track:
        ids = [i for i in ids if assignments[i].get("track") == args.track]
    problems = lint(cur, assignments, cards)
    bad = 0
    for aid in ids:
        a = need(assignments, aid)
        if a.get("kind") != "code":
            print(f"  skip  {aid} (written)")
            continue
        res = run_tests_local(read_file(a, "solution.py"), read_file(a, "tests.py"), timeout=int(a.get("timeout") or 60))
        ok = res.get("ok")
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {aid:34s} {res.get('passed', 0)}/{res.get('total', 0)} tests, "
              f"{sum(r['ms'] for r in res.get('results', []))} ms")
        if not ok and args.verbose:
            print(format_report(res))
        # Starter must import and fail (otherwise the tests don't test anything).
        if ok and args.starters:
            sres = run_tests_local(read_file(a, "starter.py"), read_file(a, "tests.py"), timeout=int(a.get("timeout") or 60))
            if sres.get("ok"):
                print(f"        WARN starter already passes every test for {aid}")
    for pr in problems:
        print("  LINT  " + pr)
    print(f"{len(ids) - bad} of {len(ids)} ok, {len(problems)} lint problems")
    sys.exit(1 if bad or problems else 0)


def lint(cur: dict, assignments: dict, cards: list[dict]) -> list[str]:
    out = []
    tracks = {t["id"]: {m["id"] for m in t.get("modules", [])} for t in cur.get("tracks", [])}
    for aid, a in assignments.items():
        d = Path(a["dir"])
        if d.name != aid:
            out.append(f"{aid}: directory name {d.name} differs from id")
        if a.get("track") not in tracks:
            out.append(f"{aid}: unknown track {a.get('track')!r}")
        elif a.get("module") not in tracks[a["track"]]:
            out.append(f"{aid}: unknown module {a.get('module')!r} for track {a['track']}")
        for f in ("README.md",) + (("starter.py", "tests.py", "solution.py") if a.get("kind") == "code" else ()):
            if not (d / f).exists():
                out.append(f"{aid}: missing {f}")
        if a.get("kind") == "code" and "def test_" not in read_file(a, "tests.py"):
            out.append(f"{aid}: tests.py has no test_ functions")
        if a.get("runtime") not in ("python", "numpy", "torch"):
            out.append(f"{aid}: runtime must be python, numpy or torch")
        if a.get("difficulty") not in ("easy", "medium", "hard"):
            out.append(f"{aid}: difficulty must be easy, medium or hard")
        if not a.get("title"):
            out.append(f"{aid}: missing title")
    seen = set()
    for c in cards:
        for k in ("id", "q", "a"):
            if not c.get(k):
                out.append(f"quiz card {c.get('id', '?')}: missing {k}")
        if c.get("id") in seen:
            out.append(f"quiz card {c['id']}: duplicate id")
        seen.add(c.get("id"))
    return out


def cmd_new(args):
    cur = load_curriculum()
    tracks = {t["id"]: {m["id"] for m in t.get("modules", [])} for t in cur.get("tracks", [])}
    if args.track not in tracks:
        sys.exit(f"track must be one of {', '.join(tracks)}")
    if args.module not in tracks[args.track]:
        sys.exit(f"module must be one of {', '.join(sorted(tracks[args.track]))}")
    d = ASSIGN / args.id
    if d.exists():
        sys.exit(f"{d} already exists")
    d.mkdir(parents=True)
    meta = {"id": args.id, "title": args.title, "track": args.track, "module": args.module, "order": args.order,
            "difficulty": args.difficulty, "minutes": args.minutes, "runtime": args.runtime, "kind": args.kind,
            "tags": [], "hints": []}
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (d / "README.md").write_text(f"# {args.title}\n\nDescribe the task, give two examples, state constraints, "
                                 "then add a short 'What the interviewer listens for' section.\n", encoding="utf-8")
    if args.kind == "code":
        (d / "starter.py").write_text('"""Starter. Keep the function names; the tests import them."""\n\n\n'
                                      "def solve():\n    raise NotImplementedError\n", encoding="utf-8")
        (d / "tests.py").write_text("from solution import solve\n\n\ndef test_example():\n    assert solve() is not None\n",
                                    encoding="utf-8")
        (d / "solution.py").write_text("def solve():\n    return 0\n", encoding="utf-8")
    print(f"Scaffolded {d}. Edit README.md, starter.py, tests.py, solution.py and meta.json, then run selfcheck.")


def cmd_sync(args):
    cur, assignments, cards, p, st = ctx()
    raw = args.data
    if raw == "-":
        raw = sys.stdin.read()
    elif os.path.exists(raw):
        raw = Path(raw).read_text(encoding="utf-8")
    raw = raw.strip()
    if not raw.startswith("[") and not raw.startswith("{"):
        raw = base64.b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8")
    data = json.loads(raw)
    events = data.get("events", []) if isinstance(data, dict) else data
    have = {e.get("id") for e in p["events"]}
    valid_types = {"start", "test", "done", "skip", "plan", "quiz", "settings"}
    n = 0
    for e in events:
        if not isinstance(e, dict) or e.get("id") in have or e.get("type") not in valid_types:
            continue
        e = {k: v for k, v in e.items() if k in ("id", "ts", "date", "type", "aid", "qid", "score", "minutes", "mode", "note",
                                                 "where", "reason", "passed", "total", "grade", "box", "due", "track",
                                                 "quiz", "settings", "deadline")}
        e.setdefault("where", "page")
        p["events"].append(e)
        have.add(e["id"])
        n += 1
    if isinstance(data, dict) and isinstance(data.get("settings"), dict):
        p["settings"].update({k: v for k, v in data["settings"].items() if k in DEFAULT_SETTINGS})
    save_progress(p)
    render_page(quiet=True)
    print(f"Merged {n} new events ({len(p['events'])} total).")


# -------------------------------------------------------------------- markdown
_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])|(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = _INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", s)
    s = _LINK.sub(lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener noreferrer">{m.group(1)}</a>', s)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _ITAL.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", s)
    return s


def md_to_html(md: str) -> str:
    """A small Markdown subset: headings, paragraphs, fenced code, lists, blockquotes, tables, inline marks."""
    out, lines, i = [], md.replace("\r\n", "\n").split("\n"), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            lang = line[3:].strip()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            code = html.escape("\n".join(lines[i + 1:j]), quote=False)
            out.append(f'<pre><code class="lang-{html.escape(lang) or "text"}">{code}</code></pre>')
            i = j + 1
            continue
        if not line.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = min(len(m.group(1)) + 1, 6)   # README's h1 becomes h2 inside the page
            out.append(f"<h{lvl}>{_inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-{2,}", lines[i + 1]):
            head = [c.strip() for c in line.strip().strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            t = "<table><thead><tr>" + "".join(f"<th>{_inline(h)}</th>" for h in head) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>"
            out.append(t)
            i = j
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.", line))
            items, j = [], i
            while j < len(lines) and lines[j].strip():
                mm = re.match(r"^\s*([-*]|\d+\.)\s+(.*)$", lines[j])
                if mm:
                    items.append(mm.group(2))
                elif items:
                    items[-1] += " " + lines[j].strip()
                else:
                    break
                j += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
            i = j
            continue
        if line.startswith(">"):
            j, quote = i, []
            while j < len(lines) and lines[j].startswith(">"):
                quote.append(lines[j][1:].strip())
                j += 1
            out.append(f"<blockquote><p>{_inline(' '.join(quote))}</p></blockquote>")
            i = j
            continue
        j, para = i, []
        while j < len(lines) and lines[j].strip() and not lines[j].startswith(("```", "#", "|", ">")) \
                and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[j]):
            para.append(lines[j].strip())
            j += 1
        if not para:
            para, j = [line.strip()], i + 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")
        i = j
    return "\n".join(out)


# --------------------------------------------------------------------- render
def page_payload() -> dict:
    cur, assignments, cards, p, st = ctx()
    order = ordered_ids(cur, assignments)
    items = []
    for tid, ids in order.items():
        for aid in ids:
            a = assignments[aid]
            items.append({
                "id": aid, "title": a.get("title", aid), "track": tid, "module": a.get("module"),
                "difficulty": a.get("difficulty"), "minutes": a.get("minutes"), "runtime": a.get("runtime"),
                "kind": a.get("kind"), "tags": a.get("tags") or [], "hints": a.get("hints") or [],
                "rubric": a.get("rubric") or [], "timeout": a.get("timeout") or 60,
                "readme": md_to_html(read_file(a, "README.md")),
                "starter": read_file(a, "starter.py"), "tests": read_file(a, "tests.py"),
                "solution": read_file(a, "solution.py"),
            })
    today = today_str()
    return {
        "meta": {"rendered": today, "rendered_at": now_iso(), "runner": RUNNER_SRC,
                 "n_assignments": len(items), "n_cards": len(cards)},
        "curriculum": cur,
        "assignments": items,
        "quiz": cards,
        "progress": {"settings": p["settings"], "events": p["events"]},
    }


def render_page(quiet: bool = False) -> None:
    if not TEMPLATE.exists():
        if not quiet:
            print(f"no template at {TEMPLATE}; skipping page render")
        return
    payload = json.dumps(page_payload(), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    out = TEMPLATE.read_text(encoding="utf-8").replace("/*__TRAIN_DATA__*/null", payload, 1)
    PAGE.parent.mkdir(exist_ok=True)
    PAGE.write_text(out, encoding="utf-8")
    if not quiet:
        print(f"wrote {PAGE} ({len(out) // 1024} KB)")


def artifact_copy(dest: Path) -> None:
    """Write the page in the shape the claude.ai Artifact tool expects: no document skeleton,
    the title, font link and stylesheet first, then the body content."""
    html_text = PAGE.read_text(encoding="utf-8")
    head = re.search(r"<head>(.*?)</head>", html_text, re.S).group(1)
    body = re.search(r"<body>(.*)</body>", html_text, re.S).group(1)
    keep = re.findall(r"<title>.*?</title>|<meta name=\"claude:[^>]*>|<link[^>]*>|<style>.*?</style>", head, re.S)
    dest.write_text("\n".join(keep) + "\n" + body.strip() + "\n", encoding="utf-8")


def cmd_render(args):
    render_page()
    if args.artifact:
        artifact_copy(Path(args.artifact))
        print(f"wrote artifact copy to {args.artifact}")


def cmd_open(args):
    render_page(quiet=True)
    webbrowser.open(PAGE.resolve().as_uri())
    print(f"Opened {PAGE}")


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.train", description="Interview training")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("today", help="today's assignment and quiz")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_today)

    s = sub.add_parser("start", help="start an assignment (copies the starter into training/work/)")
    s.add_argument("id")
    s.add_argument("--force", action="store_true", help="overwrite your work file with the starter")
    s.add_argument("--minutes", type=int, default=0, help="start a timer (mock mode)")
    s.set_defaults(fn=cmd_start)

    s = sub.add_parser("show", help="print the problem statement")
    s.add_argument("id")
    s.add_argument("--tests", action="store_true")
    s.add_argument("--solution", action="store_true")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("hint", help="show hints")
    s.add_argument("id")
    s.add_argument("n", nargs="?", type=int, default=1)
    s.set_defaults(fn=cmd_hint)

    s = sub.add_parser("test", help="run the tests against your work file")
    s.add_argument("id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_test)

    s = sub.add_parser("done", help="record an assignment as done (runs the tests first)")
    s.add_argument("id")
    s.add_argument("--minutes", type=int)
    s.add_argument("--score", type=float, help="0-100, for written assignments")
    s.add_argument("--mock", action="store_true")
    s.add_argument("--note", default="")
    s.add_argument("--force", action="store_true", help="record even if tests fail")
    s.set_defaults(fn=cmd_done)

    s = sub.add_parser("skip", help="skip an assignment for now")
    s.add_argument("id")
    s.add_argument("--reason", default="")
    s.set_defaults(fn=cmd_skip)

    s = sub.add_parser("quiz", help="today's flashcards, or record grades")
    s.add_argument("--n", type=int, default=0)
    s.add_argument("--all", action="store_true")
    s.add_argument("--json", action="store_true")
    s.add_argument("--grade", nargs="*", help="q-id=again|hard|good|easy ...")
    s.set_defaults(fn=cmd_quiz)

    s = sub.add_parser("mock", help="timed practice on a random open problem")
    s.add_argument("--track")
    s.add_argument("--minutes", type=int)
    s.add_argument("--any", action="store_true", help="include solved problems")
    s.set_defaults(fn=cmd_mock)

    s = sub.add_parser("status", help="progress summary")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("plan", help="preview the next days")
    s.add_argument("--days", type=int, default=14)
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("log", help="recent events")
    s.add_argument("--n", type=int, default=20)
    s.set_defaults(fn=cmd_log)

    s = sub.add_parser("goal", help="show or change goal, target date, rotation")
    s.add_argument("--title")
    s.add_argument("--target")
    s.add_argument("--rotation", help="7 comma-separated track ids, Monday first (review = redo weak ones)")
    s.add_argument("--quiz-per-day", type=int)
    s.set_defaults(fn=cmd_goal)

    s = sub.add_parser("selfcheck", help="run tests against reference solutions and lint the content")
    s.add_argument("ids", nargs="*")
    s.add_argument("--track")
    s.add_argument("--verbose", "-v", action="store_true")
    s.add_argument("--starters", action="store_true", help="also check that starters fail the tests")
    s.set_defaults(fn=cmd_selfcheck)

    s = sub.add_parser("new", help="scaffold an assignment")
    s.add_argument("id")
    s.add_argument("--track", required=True)
    s.add_argument("--module", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--order", type=int, default=50)
    s.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    s.add_argument("--minutes", type=int, default=30)
    s.add_argument("--runtime", default="python", choices=["python", "numpy", "torch"])
    s.add_argument("--kind", default="code", choices=["code", "written"])
    s.set_defaults(fn=cmd_new)

    s = sub.add_parser("sync", help="merge events exported from the page (file, base64 string, or - for stdin)")
    s.add_argument("data")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("render", help="rebuild docs/train.html")
    s.add_argument("--artifact", help="also write a skeleton-free copy for publishing as a claude.ai artifact")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("open", help="rebuild and open the page")
    s.set_defaults(fn=cmd_open)

    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
