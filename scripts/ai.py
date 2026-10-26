"""Claude-powered discovery (web search) and screening, with spend tracking.

Spend is estimated from the usage block on every response and recorded per
calendar month in data/state.json. Once the month's spend reaches
ai.monthly_budget_usd, every AI step is skipped until the next month.
"""
from __future__ import annotations

import datetime as dt
import os

from .common import extract_json


class AI:
    def __init__(self, cfg: dict, state: dict, log):
        import anthropic  # imported lazily so --no-ai runs need no API key

        self.cfg, self.state, self.log = cfg, state, log
        self.ai_cfg = cfg["ai"]
        self.client = anthropic.Anthropic(max_retries=3, timeout=180)
        self.month = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m")
        self.state.setdefault("spend", {}).setdefault(self.month, 0.0)
        self.run_cost = 0.0
        self.searches = 0

    # ----------------------------------------------------------- accounting
    @property
    def month_spend(self) -> float:
        return float(self.state["spend"].get(self.month, 0.0))

    def budget_left(self) -> float:
        return float(self.ai_cfg.get("monthly_budget_usd", 0)) - self.month_spend

    def can_spend(self, estimate: float = 0.05) -> bool:
        return self.budget_left() > estimate

    def _account(self, model: str, usage) -> None:
        price = self.ai_cfg["prices"].get(model) or {"input": 2.0, "output": 10.0}
        inp = getattr(usage, "input_tokens", 0) or 0
        cache_w = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_r = getattr(usage, "cache_read_input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        stu = getattr(usage, "server_tool_use", None)
        searches = (getattr(stu, "web_search_requests", 0) or 0) if stu else 0
        cost = ((inp + 1.25 * cache_w + 0.1 * cache_r) * price["input"] + out * price["output"]) / 1e6
        cost += searches * float(self.ai_cfg.get("web_search_per_request", 0.01))
        self.searches += searches
        self.run_cost += cost
        self.state["spend"][self.month] = round(self.month_spend + cost, 4)

    def _call(self, **kw):
        """messages.create with pause_turn handling (long server-tool turns)."""
        messages = list(kw.pop("messages"))
        resp = None
        for _ in range(5):
            resp = self.client.messages.create(messages=messages, **kw)
            self._account(kw["model"], resp.usage)
            if resp.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": resp.content})
        return resp

    @staticmethod
    def _text(resp) -> str:
        return "\n".join(b.text for b in (resp.content if resp else []) if getattr(b, "type", "") == "text")

    # ------------------------------------------------------------ discovery
    def discover(self, query: str) -> list[dict]:
        if not self.can_spend(0.15):
            return []
        prompt = f"""Find currently open, full-time job postings in the United States that match this focus:
{query}

Who they are for: a materials-science PhD graduating in May 2027 who works on ML for materials discovery (ML interatomic potentials, crystal structure prediction) and on LLM systems (routing, evaluation, fine-tuning).

Include only postings that are:
- full-time and based in the United States,
- aimed at PhD graduates or early-career researchers or engineers (new grad, university grad, 2027 start, or no seniority requirement beyond a PhD),
- not internships, and not at quantitative trading firms or hedge funds,
- still open (skip anything marked closed or expired).

Prefer the employer's own posting or its applicant-tracking page (Greenhouse, Lever, Ashby, Workday, company careers site) over aggregators.

Reply with only a JSON array of at most 15 objects, each:
{{"company": "...", "title": "...", "url": "...", "location": "...", "posted": "YYYY-MM-DD or empty", "note": "a few words on why it matches"}}
If nothing qualifies, reply [].""".strip()
        tool = {"type": self.ai_cfg.get("web_search_tool", "web_search_20250305"), "name": "web_search",
                "max_uses": int(self.ai_cfg.get("searches_per_query", 3)),
                "user_location": {"type": "approximate", "country": "US"}}
        try:
            resp = self._call(model=self.ai_cfg["discover_model"], max_tokens=4000,
                              messages=[{"role": "user", "content": prompt}], tools=[tool])
        except Exception as e:  # noqa: BLE001
            self.log.warn(f"discovery failed for '{query}': {e!r}")
            return []
        items = extract_json(self._text(resp))
        if not isinstance(items, list):
            self.log.warn(f"discovery returned no JSON for '{query}'")
            return []
        out = []
        for it in items[:15]:
            if not isinstance(it, dict) or not str(it.get("url", "")).startswith("http"):
                continue
            out.append({
                "company": str(it.get("company", "")).strip()[:120], "title": str(it.get("title", "")).strip()[:200],
                "url": str(it["url"]).strip(), "location": str(it.get("location", "")).strip()[:160],
                "source": f"Claude web search: {query[:60]}", "posted": str(it.get("posted", ""))[:10],
                "text": "", "pay_hint": "", "ats": None, "discovered": True,
            })
        return out

    # ------------------------------------------------------------ screening
    def screen(self, c: dict, text: str, model: str | None = None) -> dict | None:
        if not self.can_spend(0.05):
            return None
        cfg = self.cfg
        prompt = f"""You screen one job posting for a candidate who already holds a full-time offer.

Candidate: {cfg['candidate_profile']}
Their offer: about ${cfg['offer_line_k']}K total pay per year ({cfg.get('offer_description', '')}).
The candidate only wants US roles, needs an employer that sponsors work visas (H-1B), and does not want quantitative trading or hedge-fund roles.

Posting (found via {c.get('source', '')}):
Company: {c.get('company', '')}
Title: {c.get('title', '')}
Location: {c.get('location', '')}
Listed pay, if any: {c.get('pay_hint', '') or 'not given'}
URL: {c.get('url', '')}
Text:
<<<
{(text or '(page text unavailable; judge from the fields above)')[:14000]}
>>>

Estimate year-1 total pay (base plus typical bonus plus annualized equity) if this candidate were hired at the PhD new-grad level. Use the posting's own range when it has one; otherwise use what this employer typically pays PhD new grads in this kind of role. Rate fit with the candidate's background from 1 (poor) to 5 (excellent).

Reply with only this JSON object:
{{"company": "...", "role": "exact job title", "location": "...", "in_us": true, "open": true,
  "level": "phd_new_grad | early_career | senior | intern | other",
  "quant": false, "restricted": false, "sponsorship": "yes | no | unclear",
  "posted_pay": "pay as stated, or Not listed", "est_low_k": 250, "est_high_k": 320,
  "fit": 4, "track": "science | llm", "posted": "YYYY-MM-DD or empty",
  "why": "one or two sentences on why it fits this candidate, citing their specific experience",
  "caveat": "one sentence on the main risk: seniority, start date, location, or pay uncertainty"}}
Field rules: open is false if the text says the posting is closed or no longer accepting applications. restricted is true if it requires US citizenship, permanent residency, or a security clearance. sponsorship is "yes" only if the text says it sponsors visas, "no" if it says it will not sponsor or requires existing work authorization, otherwise "unclear"."""
        try:
            resp = self._call(model=model or self.ai_cfg["screen_model"], max_tokens=900,
                              messages=[{"role": "user", "content": prompt}])
        except Exception as e:  # noqa: BLE001
            self.log.warn(f"screening failed for {c.get('url')}: {e!r}")
            return None
        r = extract_json(self._text(resp))
        return r if isinstance(r, dict) else None


def ai_available(cfg: dict) -> bool:
    return bool(cfg.get("ai", {}).get("enabled")) and bool(os.environ.get("ANTHROPIC_API_KEY"))
