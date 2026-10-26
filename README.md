# Offer board, fall 2027

Full-time US roles for a May 2027 PhD, at employers that sponsor work visas, that pay about as much as the Amazon return offer or more. Quant trading roles are left out.

A GitHub Action rebuilds this board twice a day. Each run:

1. **Re-checks every role on the board.** It uses the employer's applicant-tracking record where it can (Greenhouse, Lever, Ashby, Workday), and otherwise reads the posting page. A role leaves the board after two runs in a row that say it is closed.
2. **Collects new postings from all sources.**
   - Free sources: the public job APIs of the companies in `config.yaml`, Google's careers search, NVIDIA's Workday search, and the speedyapply and Simplify new-grad trackers.
   - Claude web searches on a rotating set of queries.
3. **Drops anything already seen**, then applies cheap keyword rules (US only, no internships or managers, research or ML roles, PhD or new-grad or science signal).
4. **Screens the rest with Claude.** For each posting it estimates year-1 pay, judges fit with your CV, and checks location, level, visa terms and quant. It keeps only roles above or near your offer line.
5. **Updates the board and notifies you.** It writes `data/*.json`, the table below, and `docs/index.html`, and opens a GitHub issue listing new and closed roles. GitHub emails you about the issue.

## One-time setup

1. **Create a private repository** and push these files to it.
2. **Get an API key paid by your monthly credits.**
   - On claude.ai, open **Settings → Billing**, find **API credits**, and link a Claude Console organization. Max 20x includes $200 a month, which resets every billing cycle and doesn't roll over.
   - In the [Claude Console](https://platform.claude.com/), create a workspace just for this project and give it a spend limit, for example $120. Then create an API key in that workspace.
3. **Add the key as a repository secret** named `ANTHROPIC_API_KEY`: **Settings → Secrets and variables → Actions → New repository secret**.
4. **Run it once:** **Actions → Update offer board → Run workflow**. After that it runs on its own at about 8 AM and 8 PM Eastern.
5. **Make sure you get notified.** Check that you're watching the repository (**Watch → All activity** or **Custom → Issues**) so new-role issues reach your email.

If the commit step fails with a 403 error, open **Settings → Actions → General → Workflow permissions** and choose **Read and write permissions**.

## Running it with Claude Code instead of an API key

The repo also works with Claude Code cloud sessions and routines.
- **Rules:** `CLAUDE.md` holds the rules and the inbox format.
- **Inbox:** whatever Claude finds goes into `data/inbox/`. `python -m scripts.ingest` merges it, and every workflow run does the same.
- **Network:** first set your cloud environment's **Network access** to **Full**. The default Trusted allowlist blocks job sites.

You can use it two ways:
- **Deep sweep, started by you.** Open a cloud session on this repo at claude.ai/code and say "Follow prompts/deep-sweep.md". It adds more companies to `config.yaml`, searches every employer, checks each posting on the employer's site, and commits.
- **Daily routine, no API key.** At claude.ai/code/routines, create a routine with:
  - prompt: "Follow prompts/daily-routine.md"
  - this repository
  - a daily schedule
  - the Full-network environment
  - **Allow unrestricted branch pushes** turned on for this repo, so it can commit to `main`

  Routines run on your plan's usage. If you use a routine, disable the GitHub workflow (**Actions → Update offer board → ⋯ → Disable workflow**) so only one of them updates the board.

## Cost

The free sources and link checks cost nothing.

Claude spending is estimated from every response and stops for the month at `ai.monthly_budget_usd` in `config.yaml` (default $100). A typical run is well under $1:
- about 30 web searches at $10 per 1,000,
- Haiku 5.5 for discovery,
- Sonnet 5.5 to screen roughly 10–60 new postings.

The README header shows month-to-date spend. To spend less, lower `queries_per_run` or `max_screen_per_run`, or set `screen_model` to `claude-haiku-5-5`.

## Viewing the board

- **On GitHub:** the table below updates on every run.
- **Full interactive page:** download `docs/index.html` and open it in a browser. It has filters, the offer-line chart, and statuses and notes that save in that browser.

GitHub Pages isn't recommended here. On most plans, a Pages site from a private repository is still public on the internet, and this board includes notes about your background and visa needs.

## Tuning

Everything lives in `config.yaml`:

- **Offer line and tiers:** `offer_line_k`, `tiers`
- **Filters:** `filters`, which cover US only, visa sponsorship, quant firms, science companies and known sponsors
- **Companies whose job APIs are polled:** `sources`
- **Discovery queries:** `discovery_queries`
- **Budget and models:** `ai`

To re-screen everything after changing the rules, delete `data/seen.json` (it costs one larger run). To run locally:

```bash
pip install -r requirements.txt
python -m scripts.update --no-ai     # free sources and link checks only
ANTHROPIC_API_KEY=... python -m scripts.update
```

<!-- BOARD:START -->

**12 open roles** · 4 above your $300K line · 8 near it · last run 2026-10-25 (seeded by hand) · AI spend this month $0.00 of $100

### Above your offer line

| Company | Role | Location | Est. year-1 pay (vs offer) | Fit | H-1B | Posted |
|---|---|---|---|---|---|---|
| ByteDance Seed | [Research Scientist, Quantum Chemistry (Global Frontier Tech Program, 2027 start, PhD)](https://joinbytedance.com/search/7632426574734035253) | San Jose, CA | $300K–$450K (+25%) | ●●●●● | sponsors | Open now (2027-start req) |
| Periodic Labs | [Research Scientist / Research Engineer, Materials Data](https://jobs.a16z.com/jobs/periodic-labs/research-scientist-research-engineer-materials-data--ats-mn7ddw8dg6yjkd5er8t6hxq99x8e6krx) | Menlo Park, CA | $280K–$420K (+17%) | ●●●●● | says it sponsors | Sep 1, 2026 |
| ByteDance Seed | [Research Engineer Graduate, Seed Model AI Foundation Model Infrastructure (2027 start, PhD)](https://portfoliojobs.tcv.com/companies/bytedance/jobs/72958826-research-scientist-graduate-seed-ai-foundation-model-infrastructure-2027-start-phd) | Seattle, WA | $300K–$420K (+20%) | ●●● | sponsors | Spring 2026, still taking applications |
| Periodic Labs | [Research Scientist / Research Engineer, Midtraining](https://jobs.a16z.com/jobs/periodic-labs/research-scientist-research-engineer-midtraining--ats-mn783xrp2jmjq4cp3mdjvsjbp98fc7j7) | Menlo Park, CA | $280K–$420K (+17%) | ●●● | says it sponsors | About Sep 30, 2026 |

### Near your offer line

| Company | Role | Location | Est. year-1 pay (vs offer) | Fit | H-1B | Posted |
|---|---|---|---|---|---|---|
| Meta FAIR | [Research Engineer, AI for Chemistry](https://metacareers.com/v2/jobs/3846804338907958) | Seattle, WA | $250K–$340K (-2%) | ●●●●● | sponsors | Live on Meta’s site |
| Google DeepMind | [Research Engineer, Materials Science](https://job-boards.greenhouse.io/deepmind/jobs/7339890) | Mountain View, CA | $230K–$320K (-8%) | ●●●●● | sponsors | May 2026, still taking applications |
| Radical AI | [AI Research Scientist, GNN / Foundation Models](https://jobs.lever.co/RadicalAI/025709b9-d365-4909-aecb-aba5669703da) | New York, NY | $210K–$320K (-12%) | ●●●●● | **ask first** | Live on Radical’s Lever board |
| TikTok | [Machine Learning Scientist Graduate, Trust and Safety (2027 start, PhD)](https://lifeattiktok.com/search/7665985896410368261) | Seattle, WA | $255K–$330K (-3%) | ●●●● | sponsors | About Aug 2026 |
| Lila Sciences | [Research Scientist, Computational Condensed Matter Physics](https://jobs.generalcatalyst.com/companies/lila-sciences-2/jobs/87897567-research-scientist-computational-condensed-matter-physics) | Cambridge, MA | $220K–$330K (-8%) | ●●●● | sponsors | Jul 29, 2026 |
| NVIDIA | [Research Scientist, Fundamental Generative AI (new college grad)](https://jobs.anitab.org/companies/nvidia/jobs/95232560-research-scientist-fundamental-generative-ai-new-college-grad-2026) **New** | Santa Clara, CA | $270K–$360K (+5%) | ●●● | sponsors | Sep 30, 2026 |
| Meta | [Research Scientist, Machine Learning (PhD), university grad](https://www.metacareers.com/students-and-grads) | Sunnyvale, CA and 7 more | $250K–$320K (-5%) | ●●● | sponsors | Listed on Meta’s Students & Grads page |
| Google | [Software Engineer, AI/Machine Learning, PhD, Early Career (2027 start)](https://www.google.com/about/careers/applications/jobs/results/123087196289671878-software-engineer) | Sunnyvale, Seattle, Kirkland, New York and 6 more | $240K–$310K (-8%) | ●●● | sponsors | Aug 2026 |

<details><summary>Screened out (15)</summary>

- [Quant and trading roles (Jane Street, Citadel, Citadel Securities, SIG, Hudson River Trading)](https://www.tradermath.org/salaries/quant-researcher): Left off the board by request.
- [Meta FAIR, AI Research Scientist, FAIR Chemistry (New York)](https://jobs.accel.com/companies/facebook-2-07d2eb1b-d3ad-4084-93df-74a124ac9233/jobs/86456920-ai-research-scientist-fair-chemistry): Closed: the posting stopped accepting applications.
- [Lila Sciences, ML Research Scientist I/II, Multimodal Data Extraction](https://startup.jobs/ml-research-scientist-i-ii-multimodal-data-extraction-lila-sciences-7844988): Closed: no longer accepting applications.
- [NVIDIA, AI Scientist, Agentic AI for Science (new grad)](https://inferencejobs.com/companies/nvidia/jobs/ai-scientist-agentic-ai-for-science-phd-new-college-grad-2026): Couldn’t confirm it’s still open: applications were only guaranteed until Feb 14, 2026, and its senior sibling has closed.
- [Periodic Labs, Research Scientist, Condensed Matter Theory](https://www.startuphub.ai/jobs/periodic-labs/research-scientist-condensed-matter-theory-db3a63): Couldn’t confirm it on Periodic’s own board, and it wants a condensed-matter theory PhD.
- [TikTok, LLM Post-Training Engineer Graduate (2027 start)](https://lifeattiktok.com/search/7673672141012093189): Still open, but it’s a master’s-level req, so it would likely be leveled below an L5 PhD offer.
- [TikTok, Business Integrity research scientist and E-commerce LLM/agent engineer (2027 start, PhD)](https://www.themuse.com/jobs/tiktok/research-scientist-graduate-business-integrity-2027-start-phd): Couldn’t confirm them on TikTok’s own site. With the two-application cap, the confirmed ByteDance and TikTok roles above are better uses of your slots.
- [Microsoft, Applied Scientist II (AI Development Acceleration Program)](https://jobs.anitab.org/companies/microsoft/jobs/56602222-applied-scientist-ii-microsoft-ai-development-acceleration-program-redmond): Posted base $100.6K–$199K. Microsoft’s new-grad applied-scientist packages typically land well under an Amazon L5 package.
- [Microsoft Research AI for Science, materials roles (MatterGen and MatterSim team)](https://jobs.anitab.org/companies/microsoft/jobs/81414595-ai-for-science-residency-machine-learning-resident): Outside the US (Cambridge, UK and Berlin), and UK and EU pay converts to well below a Seattle offer.
- [Google DeepMind, London materials roles (Material Intelligence; Semiconductor Materials)](https://job-boards.greenhouse.io/deepmind/jobs/7353237): Outside the US (London), and the semiconductor role asks for 2+ years leading a research agenda.
- [Periodic Labs, Multiphysics Simulation Scientist and Forward Deployed Engineer (Physics & Simulation)](https://www.inferencejobs.com/companies/periodic-labs): Base $160K–$220K and $180K–$250K.
- [Lila Sciences, Scientist II and Computational Scientist I/II roles](https://startup.jobs/computational-scientist-i-ii-soft-matter-formulations-solids-and-melts-lila-sciences-8807717): Base roughly $118K–$198K.
- [Argonne National Laboratory, AI and autonomous-lab materials scientist](https://www.careercircle.com/jobs/all/all/usa/il/lemont/fcaef6f9-4a20-4973-8e36-611a2a297001): Base $116K–$181K.
- [OpenAI Residency](https://openai.com/residency): $18.3K a month for a six-month program, and the 2026 cycle is closed. Worth it only as a pivot into frontier-lab research.
- [NVIDIA ALCHEMI senior roles (chemistry and materials software)](https://inferencejobs.com/companies/nvidia/jobs/senior-machine-learning-engineer-chemistry-and-materials-science): Pay is fine, but they ask for 5–8+ years of experience. Revisit after a couple of years in industry.

</details>

<!-- BOARD:END -->
