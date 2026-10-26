"""Shared helpers: config/state IO, HTTP, text extraction and rule-based filters."""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0 Safari/537.36 offer-board/1.0")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})


# --------------------------------------------------------------------------- IO
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def days_between(a: str, b: str) -> int:
    try:
        return (dt.date.fromisoformat(b[:10]) - dt.date.fromisoformat(a[:10])).days
    except (TypeError, ValueError):
        return 10**6


# ------------------------------------------------------------------------- HTTP
def http_get(url: str, timeout: int = 25) -> requests.Response | None:
    try:
        return SESSION.get(url, timeout=timeout, allow_redirects=True)
    except requests.RequestException:
        return None


def http_json(url: str, method: str = "GET", payload=None, timeout: int = 25):
    """Return (parsed_json_or_None, status_code). status 0 means a network error."""
    try:
        r = SESSION.request(method, url, json=payload, timeout=timeout,
                            headers={"Accept": "application/json"})
    except requests.RequestException:
        return None, 0
    if r.status_code != 200:
        return None, r.status_code
    try:
        return r.json(), 200
    except ValueError:
        return None, r.status_code


# ------------------------------------------------------------------------- text
_DROP_BLOCKS = re.compile(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>")
_BREAKS = re.compile(r"(?i)<br\s*/?>|</p>|</li>|</h[1-6]>|</div>|</tr>")
_TAGS = re.compile(r"<[^>]+>")


def html_to_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s) if "&lt;" in s[:2000] else s  # Greenhouse escapes its HTML
    s = _DROP_BLOCKS.sub(" ", s)
    s = _BREAKS.sub("\n", s)
    s = _TAGS.sub(" ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v\u00a0]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def page_title(raw_html: str) -> str:
    m = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', raw_html or "", re.I)
    if not m:
        m = re.search(r"<title[^>]*>(.*?)</title>", raw_html or "", re.I | re.S)
    return html.unescape(m.group(1)).strip() if m else ""


_TRACKING = {"gh_src", "source", "ref", "src", "utm_source", "utm_medium", "utm_campaign",
             "utm_content", "utm_term", "lever-source", "lever-origin", "gclid", "trk"}


def norm_url(url: str) -> str:
    if not url:
        return ""
    p = urlparse(url.strip())
    query = [(k, v) for k, v in parse_qsl(p.query) if k.lower() not in _TRACKING]
    path = re.sub(r"/application/?$", "", p.path).rstrip("/")
    host = p.netloc.lower().removeprefix("www.")
    return urlunparse(("https", host, path, "", urlencode(query), ""))


def job_key(company: str, title: str, url: str) -> str:
    u = norm_url(url)
    if u:
        return u
    return "t:" + hashlib.sha1(f"{company}|{title}".lower().encode()).hexdigest()


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:60]


# --------------------------------------------------------------------- location
_ABBR = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV "
         "NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()
_STATES = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
           "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
           "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
           "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
           "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
           "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
           "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
           "washington", "west virginia", "wisconsin", "wyoming"]
_US_CITIES = ["san francisco", "san jose", "sunnyvale", "mountain view", "menlo park", "palo alto",
              "santa clara", "redwood city", "san mateo", "san bruno", "seattle", "bellevue",
              "redmond", "kirkland", "boston", "cambridge, ma", "pittsburgh", "austin",
              "chicago", "los angeles", "culver city", "san diego", "irvine", "atlanta",
              "raleigh", "durham", "madison", "denver", "boulder", "princeton", "philadelphia",
              "miami", "houston", "dallas", "salt lake", "portland", "ann arbor", "burlingame",
              "brooklyn", "manhattan", "jersey city", "bay area", "silicon valley"]
_US_WORDS = re.compile(r"united states|\busa\b|\bu\.s\.|\bus\b|\bus-[a-z]{2}\b|remote[^a-z]*(us|united states)", re.I)
_ABBR_RE = re.compile(r"(?:,|\b)\s*(%s)\b(?!-)" % "|".join(_ABBR))
_NON_US = ["london", "united kingdom", "england", ", uk", " uk ", "canada", "toronto", "montreal",
           "montréal", "vancouver", "singapore", "beijing", "shanghai", "shenzhen", "hong kong",
           "taipei", "taiwan", "japan", "tokyo", "seoul", "korea", "india", "bangalore",
           "bengaluru", "hyderabad", "berlin", "germany", "munich", "paris", "france", "zurich",
           "zürich", "switzerland", "amsterdam", "netherlands", "dublin", "ireland", "israel",
           "tel aviv", "sydney", "australia", "mexico", "brazil", "poland", "warsaw", "armenia",
           "yerevan", "spain", "madrid", "italy", "sweden", "denmark", "finland", "norway",
           "china", "jakarta", "dubai", "cambridge, uk", "oxford", "kraków", "prague", "vietnam"]


def is_us(location: str) -> bool | None:
    """True if any listed location is in the US, False if clearly not, None if unknown."""
    if not location:
        return None
    loc = " " + location + " "
    low = loc.lower()
    if (_US_WORDS.search(loc) or _ABBR_RE.search(loc) or any(s in low for s in _STATES)
            or any(c in low for c in _US_CITIES)):
        return True
    if any(n in low for n in _NON_US):
        return False
    if "remote" in low:
        return None
    return None


# ------------------------------------------------------------------ title rules
RELEVANT = re.compile(r"research|scientist|machine learning|\bml\b|\bai\b|artificial intelligence|"
                      r"deep learning|\bllm|applied scien|foundation model|generative|computational|"
                      r"materials|chemi|atomistic|simulation|data scien", re.I)
NEWGRAD = re.compile(r"ph\.?\s?d|new (college )?grad|university grad|early career|graduate|2027|"
                     r"campus|entry[- ]level|\bfresh|residency|resident\b", re.I)
SCIENCE = re.compile(r"material|chemi|molecul|atomistic|crystal|quantum|physic|scien(ce|tific)|"
                     r"drug|protein|biolog|catal|battery|energy|semiconductor|interatomic", re.I)
EXCLUDE = re.compile(r"\b(intern|internship|co-?op|director|head of|manager|principal|(?<!technical )staff|"
                     r"technician|recruit\w*|sales|marketing|account (executive|manager)|legal|"
                     r"counsel|finance|controller|people|operations|facilit\w*|ehs|safety|"
                     r"mechanical|electrical|product manager|program manager|designer|analyst|"
                     r"administrator|coordinator|associate director|vp|chief|postdoc\w*|post-doctoral|"
                     r"fellow(ship)?|clear(ed|ance)|ts/sci|bs/ms|master'?s|bachelor'?s?|undergrad\w*|"
                     r"leadership development|rotation(al)?|apprentice\w*|hardware|asic|vlsi|rtl|"
                     r"firmware|embedded|sdet|test engineer|devops|site reliability|support engineer|"
                     r"solutions? (architect|engineer)|architect|consultant)\b", re.I)
SENIOR = re.compile(r"\b(senior|sr\.?)\b|^lead\b|\blead (scientist|engineer|researcher|ml|ai)\b", re.I)
QUANT_TITLE = re.compile(r"quant(itative)?\s+(trader|trading|research|researcher|developer|analyst)|"
                         r"\btrading\b|algorithm develop", re.I)


def _in_list(company: str, names) -> bool:
    c = (company or "").lower()
    return any(n.lower() == c or n.lower() in c for n in names or [])


def is_quant(company: str, title: str, cfg: dict) -> bool:
    return bool(QUANT_TITLE.search(title or "")) or _in_list(company, cfg["filters"]["quant_companies"])


def prefilter(c: dict, cfg: dict) -> tuple[bool, str]:
    """Cheap rules applied before any paid screening. Returns (keep, reason)."""
    title, company = c.get("title", ""), c.get("company", "")
    f = cfg["filters"]
    if _in_list(company, f.get("exclude_companies")):
        return False, "excluded company"
    if f.get("exclude_quant") and (c.get("quant") or is_quant(company, title, cfg)):
        return False, "quant or trading"
    if EXCLUDE.search(title):
        return False, "title outside scope (intern, manager, non-research, ...)"
    if f.get("us_only") and is_us(c.get("location", "")) is False:
        return False, "outside the US"
    if f.get("require_visa_sponsorship") and c.get("sponsorship_hint") in ("no", "citizens_only"):
        return False, "says no sponsorship or requires citizenship"
    if not RELEVANT.search(title):
        return False, "not a research or ML role"
    science_co = _in_list(company, f.get("science_companies"))
    if SENIOR.search(title) and not science_co:
        return False, "senior title"
    if not (NEWGRAD.search(title) or science_co or SCIENCE.search(title) or c.get("phd")):
        return False, "no PhD/new-grad or science signal"
    return True, ""


def known_sponsor(company: str, cfg: dict) -> bool:
    return _in_list(company, cfg["filters"].get("known_sponsors"))


def extract_json(text: str):
    """Parse the first JSON object/array in a model reply."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None
