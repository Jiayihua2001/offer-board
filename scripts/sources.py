"""Deterministic job sources: applicant-tracking APIs, career-site searches and GitHub trackers.

Every collector returns a list of candidate dicts:
  company, title, url, location, source, posted (YYYY-MM-DD or ""), text (may be ""),
  pay_hint, ats (dict for reliable open/closed checks, or None), plus optional flags.
"""
from __future__ import annotations

import datetime as dt
import html
import re
from urllib.parse import quote

from .common import SESSION, html_to_text, http_get, http_json, page_title


def _date_from_ms(ms) -> str:
    try:
        return dt.datetime.fromtimestamp(int(ms) / 1000, dt.timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return ""


# ---------------------------------------------------------------- Greenhouse
def greenhouse(token: str, company: str, log) -> list[dict]:
    data, code = http_json(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true")
    if data is None:
        log.warn(f"Greenhouse board '{token}' unavailable (HTTP {code})")
        return []
    out = []
    for j in data.get("jobs", []):
        out.append({
            "company": company, "title": j.get("title", ""), "url": j.get("absolute_url", ""),
            "location": (j.get("location") or {}).get("name", ""),
            "source": f"Greenhouse ({token})", "posted": (j.get("first_published") or j.get("updated_at") or "")[:10],
            "text": html_to_text(j.get("content", "")), "pay_hint": "",
            "ats": {"type": "greenhouse", "token": token, "id": str(j.get("id"))},
        })
    return out


# --------------------------------------------------------------------- Lever
def lever(site: str, company: str, log) -> list[dict]:
    data, code = http_json(f"https://api.lever.co/v0/postings/{site}?mode=json")
    if data is None:
        log.warn(f"Lever site '{site}' unavailable (HTTP {code})")
        return []
    out = []
    for j in data:
        cats = j.get("categories") or {}
        lists = "\n".join(f"{x.get('text', '')}\n{html_to_text(x.get('content', ''))}" for x in j.get("lists") or [])
        sal = j.get("salaryRange") or {}
        pay = ""
        if sal.get("min") or sal.get("max"):
            pay = f"{sal.get('currency', '')} {sal.get('min', '?')}–{sal.get('max', '?')} {sal.get('interval', '')}".strip()
        out.append({
            "company": company, "title": j.get("text", ""), "url": j.get("hostedUrl", ""),
            "location": cats.get("location") or ", ".join(cats.get("allLocations") or []),
            "source": f"Lever ({site})", "posted": _date_from_ms(j.get("createdAt")),
            "text": "\n".join([j.get("descriptionPlain", ""), lists, j.get("additionalPlain", "")]),
            "pay_hint": pay, "ats": {"type": "lever", "site": site, "id": j.get("id", "")},
        })
    return out


# --------------------------------------------------------------------- Ashby
def ashby_jobs(org: str):
    data, code = http_json(f"https://api.ashbyhq.com/posting-api/job-board/{org}?includeCompensation=true")
    return (data or {}).get("jobs"), code


def ashby(org: str, company: str, log) -> list[dict]:
    jobs, code = ashby_jobs(org)
    if jobs is None:
        log.warn(f"Ashby board '{org}' unavailable (HTTP {code})")
        return []
    out = []
    for j in jobs:
        if j.get("isListed") is False:
            continue
        comp = j.get("compensation") or {}
        out.append({
            "company": company, "title": j.get("title", ""), "url": j.get("jobUrl", ""),
            "location": j.get("location", "") or "",
            "source": f"Ashby ({org})", "posted": (j.get("publishedAt") or "")[:10],
            "text": j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml", "")),
            "pay_hint": comp.get("compensationTierSummary") or comp.get("scrapeableCompensationSalarySummary") or "",
            "ats": {"type": "ashby", "org": org, "id": j.get("id", "")},
        })
    return out


# ------------------------------------------------------------------- Workday
def workday(spec: dict, log) -> list[dict]:
    host, tenant, site = spec["host"], spec["tenant"], spec["site"]
    api = f"https://{host}/wday/cxs/{tenant}/{site}/jobs"
    seen, out = set(), []
    for q in spec.get("searches", []):
        for offset in (0, 20, 40):
            data, code = http_json(api, "POST", {"appliedFacets": {}, "limit": 20, "offset": offset, "searchText": q})
            if data is None:
                log.warn(f"Workday {spec['company']} search '{q}' failed (HTTP {code})")
                break
            posts = data.get("jobPostings") or []
            for p in posts:
                path = p.get("externalPath", "")
                if not path or path in seen:
                    continue
                seen.add(path)
                out.append({
                    "company": spec["company"], "title": p.get("title", ""),
                    "url": f"https://{host}/en-US/{site}{path}", "location": p.get("locationsText", ""),
                    "source": f"Workday ({spec['company']})", "posted": "", "text": "", "pay_hint": "",
                    "ats": {"type": "workday", "host": host, "tenant": tenant, "site": site, "path": path},
                })
            if len(posts) < 20:
                break
    return out


def workday_detail(ats: dict) -> dict | None:
    data, code = http_json(f"https://{ats['host']}/wday/cxs/{ats['tenant']}/{ats['site']}{ats['path']}")
    if data is None:
        return {"_status": code}
    info = data.get("jobPostingInfo") or {}
    return {"_status": 200, "title": info.get("title", ""), "text": html_to_text(info.get("jobDescription", "")),
            "location": info.get("location", ""), "posted": info.get("startDate", "") or "",
            "can_apply": info.get("canApply", True)}


# ------------------------------------------------------------ Google careers
_GOOGLE_ID = re.compile(r"jobs/results/(\d{6,})-([a-z0-9-]+)")


def google_careers(queries: list[str], log) -> list[dict]:
    out, seen = [], set()
    for q in queries:
        url = ("https://www.google.com/about/careers/applications/jobs/results?"
               f"q={quote(q)}&location={quote('United States')}")
        r = http_get(url)
        if r is None or r.status_code != 200:
            log.warn(f"Google careers search '{q}' failed")
            continue
        for jid, slug in _GOOGLE_ID.findall(r.text):
            if jid in seen:
                continue
            seen.add(jid)
            out.append({
                "company": "Google", "title": slug.replace("-", " "),
                "url": f"https://www.google.com/about/careers/applications/jobs/results/{jid}-{slug}",
                "location": "United States", "source": "Google careers search", "posted": "",
                "text": "", "pay_hint": "", "ats": None, "needs_detail": True,
            })
    return out


# --------------------------------------------------------- GitHub trackers
_ROW_LINK = re.compile(r'href="([^"]+)"')
_STRONG = re.compile(r"<strong>(.*?)</strong>|\*\*(.*?)\*\*")


def speedyapply(log) -> list[dict]:
    r = http_get("https://raw.githubusercontent.com/speedyapply/2027-AI-College-Jobs/main/NEW_GRAD_USA.md", timeout=40)
    if r is None or r.status_code != 200:
        log.warn("speedyapply new-grad list unavailable")
        return []
    out, section = [], ""
    for line in r.text.splitlines():
        if line.startswith("### "):
            section = line[4:].strip().lower()
            continue
        if not line.startswith("| ") or "href" not in line:
            continue
        cols = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cols) < 4:
            continue
        m = _STRONG.search(cols[0])
        company = html.unescape(re.sub(r"<[^>]+>", "", (m.group(1) or m.group(2)) if m else cols[0])).strip()
        links = [u for c in cols[1:] for u in _ROW_LINK.findall(c)]
        if not links:
            continue
        has_salary = len(cols) >= 6
        age = re.match(r"(\d+)\s*d", cols[-1] or "")
        posted = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=int(age.group(1)))).isoformat() if age else ""
        out.append({
            "company": company, "title": html.unescape(re.sub(r"<[^>]+>", "", cols[1])),
            "url": html.unescape(links[-1]), "location": re.sub(r"<[^>]+>", "", cols[2]),
            "source": "speedyapply 2027 AI new-grad list", "posted": posted,
            "text": "", "pay_hint": cols[3] if has_salary else "", "ats": None,
            "quant": section.startswith("quant"), "age": cols[-1],
        })
    return out


_SPONSOR_MAP = {"Does Not Offer Sponsorship": "no", "U.S. Citizenship is Required": "citizens_only",
                "Offers Sponsorship": "yes"}


def simplify(log, max_age_days: int = 120) -> list[dict]:
    url = "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/.github/scripts/listings.json"
    try:
        r = SESSION.get(url, timeout=90)
        rows = r.json() if r.status_code == 200 else None
    except Exception:  # noqa: BLE001 - any failure just skips this source
        rows = None
    if rows is None:
        log.warn("Simplify new-grad list unavailable")
        return []
    cutoff = dt.datetime.now(dt.timezone.utc).timestamp() - max_age_days * 86400
    out = []
    for x in rows:
        if not x.get("active") or x.get("is_visible") is False or (x.get("date_posted") or 0) < cutoff:
            continue
        degrees = [d.lower() for d in x.get("degrees") or []]
        title = x.get("title", "")
        phd = any("phd" in d for d in degrees) or bool(re.search(r"ph\.?\s?d", title, re.I))
        if x.get("category") not in ("AI/ML/Data", "Data Science, AI & Machine Learning") and not phd:
            continue
        out.append({
            "company": x.get("company_name", ""), "title": title, "url": x.get("url", ""),
            "location": "; ".join(x.get("locations") or []), "source": "Simplify new-grad list",
            "posted": _date_from_ms((x.get("date_posted") or 0) * 1000), "text": "", "pay_hint": "",
            "ats": None, "phd": phd, "quant": x.get("category") == "Quant",
            "sponsorship_hint": _SPONSOR_MAP.get(x.get("sponsorship", ""), ""),
        })
    return out


# ------------------------------------------------------------------ helpers
def fetch_posting_text(c: dict) -> tuple[str, str]:
    """Return (text, title) for a candidate, using its ATS detail API when possible."""
    ats = c.get("ats") or {}
    if c.get("text") and len(c["text"]) > 400:
        return c["text"], c.get("title", "")
    if ats.get("type") == "workday":
        d = workday_detail(ats) or {}
        if d.get("_status") == 200:
            return d.get("text", ""), d.get("title") or c.get("title", "")
    r = http_get(c.get("url", ""))
    if r is None or r.status_code != 200:
        return "", c.get("title", "")
    return html_to_text(r.text)[:40000], page_title(r.text) or c.get("title", "")


def collect(cfg: dict, log) -> list[dict]:
    src = cfg["sources"]
    found: list[dict] = []
    plan = []
    plan += [(f"greenhouse:{g['token']}", lambda g=g: greenhouse(g["token"], g["company"], log)) for g in src.get("greenhouse", [])]
    plan += [(f"lever:{l['site']}", lambda l=l: lever(l["site"], l["company"], log)) for l in src.get("lever", [])]
    plan += [(f"ashby:{a['org']}", lambda a=a: ashby(a["org"], a["company"], log)) for a in src.get("ashby", [])]
    plan += [(f"workday:{w['company']}", lambda w=w: workday(w, log)) for w in src.get("workday", [])]
    if src.get("google_careers"):
        plan.append(("google careers", lambda: google_careers(src["google_careers"].get("queries", []), log)))
    if src.get("speedyapply"):
        plan.append(("speedyapply", lambda: speedyapply(log)))
    if src.get("simplify"):
        plan.append(("simplify", lambda: simplify(log)))
    for name, fn in plan:
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001 - one broken source must not stop the run
            log.warn(f"source {name} crashed: {e!r}")
            got = []
        log.info(f"source {name}: {len(got)} postings")
        found.extend(got)
    return found
