"""Decide whether a posting is still open.

Returns "open", "closed" or "unknown". Applicant-tracking APIs give definite
answers; ordinary pages count as closed only on a 404/410 or an explicit
"no longer accepting applications"-style message, so a slow or bot-blocking
site never knocks a live role off the board.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from .common import html_to_text, http_get, http_json
from .sources import ashby_jobs, workday_detail

CLOSED_RE = re.compile(
    r"no longer accepting applications|(job|position|posting|role) is no longer (available|open|active)|"
    r"this job is no longer|no longer available and isn.t accepting|position has been filled|"
    r"this (job|posting|position) has (closed|expired|been closed)|job (posting )?not found|"
    r"posting (has )?closed|this position is closed|job closed|requisition is no longer|"
    r"the job you are looking for is no longer|this job has expired", re.I)

_ashby_cache: dict[str, set | None] = {}


def infer_ats(url: str) -> dict | None:
    if not url:
        return None
    p = urlparse(url)
    host, parts = p.netloc.lower(), [x for x in p.path.split("/") if x]
    if "greenhouse.io" in host and "jobs" in parts:
        i = parts.index("jobs")
        if i >= 1 and i + 1 < len(parts) and parts[i + 1].isdigit():
            return {"type": "greenhouse", "token": parts[i - 1], "id": parts[i + 1]}
    if host == "jobs.lever.co" and len(parts) >= 2:
        return {"type": "lever", "site": parts[0], "id": parts[1]}
    if host == "jobs.ashbyhq.com" and len(parts) >= 2:
        return {"type": "ashby", "org": parts[0], "id": parts[1]}
    m = re.match(r"([a-z0-9-]+)\.wd\d+\.myworkdayjobs\.com", host)
    if m and "job" in parts:
        site_i = 1 if re.fullmatch(r"[a-z]{2}-[A-Z]{2}", parts[0]) else 0
        site = parts[site_i]
        path = "/" + "/".join(parts[site_i + 1:])
        return {"type": "workday", "host": host, "tenant": m.group(1), "site": site, "path": path}
    return None


def check_ats(ats: dict) -> str:
    t = ats.get("type")
    if t == "greenhouse":
        _, code = http_json(f"https://boards-api.greenhouse.io/v1/boards/{ats['token']}/jobs/{ats['id']}")
        return "open" if code == 200 else ("closed" if code == 404 else "unknown")
    if t == "lever":
        _, code = http_json(f"https://api.lever.co/v0/postings/{ats['site']}/{ats['id']}")
        return "open" if code == 200 else ("closed" if code == 404 else "unknown")
    if t == "ashby":
        org = ats["org"]
        if org not in _ashby_cache:
            jobs, _ = ashby_jobs(org)
            _ashby_cache[org] = None if jobs is None else {j.get("id") for j in jobs if j.get("isListed") is not False}
        ids = _ashby_cache[org]
        if ids is None:
            return "unknown"
        return "open" if ats["id"] in ids else "closed"
    if t == "workday":
        d = workday_detail(ats) or {}
        if d.get("_status") == 200:
            return "open" if d.get("can_apply", True) else "closed"
        return "closed" if d.get("_status") in (404, 410) else "unknown"
    return "unknown"


def check_page(url: str) -> str:
    r = http_get(url)
    if r is None:
        return "unknown"
    if r.status_code in (404, 410):
        return "closed"
    if r.status_code != 200:
        return "unknown"
    text = html_to_text(r.text)
    if CLOSED_RE.search(text[:60000]):
        return "closed"
    return "open" if len(text) > 800 else "unknown"


def check_job(job: dict) -> str:
    ats = job.get("ats") or infer_ats(job.get("url", ""))
    if ats:
        verdict = check_ats(ats)
        if verdict != "unknown":
            return verdict
    return check_page(job.get("url", ""))
