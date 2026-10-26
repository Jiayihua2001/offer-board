"""Shared screening rules: identity/dedupe helpers and the add-or-screen decision."""
from __future__ import annotations

import hashlib
import re

from . import render, verify
from .common import _in_list, is_quant, known_sponsor, slugify

_PHD = re.compile(r"[\s(\-–]*\bph\.?\s?d\b[)\s]*", re.I)


def tkey(company: str, title: str) -> str:
    full = lambda x: re.sub(r"[^a-z0-9]+", "-", (x or "").lower()).strip("-")  # noqa: E731 - untruncated slug
    return f"{full(company)}|{full(title)}"


_STOP = {"new", "college", "grad", "graduate", "graduates", "2025", "2026", "2027", "start", "phd", "the",
         "and", "of", "a", "for", "in", "university"}


def _tokens(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (title or "").lower()) if w not in _STOP}


def same_role(company_a: str, title_a: str, company_b: str, title_b: str) -> bool:
    ca, cb = slugify(company_a), slugify(company_b)
    if not ca or not cb or not (ca in cb or cb in ca):
        return False
    a, b = _tokens(title_a), _tokens(title_b)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.85


def ident(c: dict) -> str | None:
    """Stable identity from the applicant-tracking record or a requisition number in the URL."""
    ats = c.get("ats") or verify.infer_ats(c.get("url", ""))
    if ats:
        return f"{ats['type']}:{ats.get('id') or ats.get('path', '').rsplit('_', 1)[-1]}".lower()
    m = re.search(r"\b(JR\d{6,8}|R\d{6,8})\b", c.get("url", ""))
    return m.group(1).lower() if m else None


def _num(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def decide(r: dict, c: dict, cfg: dict, run_date: str):
    """Turn a screening result into ('added' | 'screened', reason, job_or_None)."""
    f = cfg["filters"]
    company = (r.get("company") or c.get("company") or "").strip()
    role = (r.get("role") or c.get("title") or "").strip()
    lo, hi = _num(r.get("est_low_k")), _num(r.get("est_high_k"))
    if lo > hi:
        lo, hi = hi, lo
    fit = int(_num(r.get("fit"), 3))
    fit = min(5, max(1, fit))
    if r.get("open") is False:
        return "screened", "posting is closed", None
    if f.get("us_only") and r.get("in_us") is False:
        return "screened", "outside the US", None
    if f.get("exclude_quant") and (r.get("quant") is True or is_quant(company, role, cfg)):
        return "screened", "quant or trading role", None
    if f.get("require_visa_sponsorship") and r.get("restricted") is True:
        return "screened", "requires US citizenship, a green card or a clearance", None
    if f.get("require_visa_sponsorship") and r.get("sponsorship") == "no":
        return "screened", "says it won’t sponsor visas", None
    level = r.get("level", "")
    stretch = False
    if "intern" in level:
        return "screened", "internship", None
    if level in ("senior", "other"):
        if _in_list(company, f.get("science_companies")) and fit >= 4:
            stretch = True
        else:
            return "screened", "needs more experience than a new PhD", None
    if not (30 <= lo <= 3000 and 30 <= hi <= 3000):
        return "screened", "couldn’t estimate pay", None
    job = {"company": company, "role": role, "est": [round(lo), round(hi)]}
    t, _ = render.tier(job, cfg)
    if t == "below":
        return "screened", f"estimated ${round(lo)}K–${round(hi)}K, below your line", None
    sponsorship = r.get("sponsorship")
    if sponsorship == "yes":
        visa = ["stated", "The posting says it sponsors visas."]
    elif known_sponsor(company, cfg):
        visa = ["yes", "Files H-1B petitions regularly (public Department of Labor data)."]
    else:
        visa = ["ask", "The posting doesn’t say; ask the recruiter."]
    posted_pay = str(r.get("posted_pay") or "").strip()
    if not posted_pay or posted_pay.lower().startswith("not listed"):
        posted_pay = c.get("pay_hint") or "Pay not listed in the posting"
    track = r.get("track") if r.get("track") in ("science", "llm") else "llm"
    digest = hashlib.sha1(c["url"].encode()).hexdigest()[:6]
    job.update({
        "id": f"{slugify(company)[:24]}-{slugify(role)[:32]}-{digest}", "track": track,
        "location": str(r.get("location") or c.get("location") or "").strip()[:120],
        "base": posted_pay[:160], "fit": fit,
        "why": str(r.get("why") or "")[:600], "caveat": str(r.get("caveat") or "")[:400],
        "posted": str(r.get("posted") or c.get("posted") or "")[:10], "source": c.get("source", ""),
        "url": c["url"], "visa": visa, "ats": c.get("ats") or verify.infer_ats(c["url"]),
        "added": run_date, "first_seen": run_date, "last_checked": run_date, "last_open": run_date,
        "check_failures": 0,
    })
    if stretch:
        job["stretch"] = True
    return "added", "", job


