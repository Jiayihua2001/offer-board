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

**16 open roles** · 5 above your $300K line · 11 near it · last run 2026-10-26 18:49 UTC · AI spend this month $0.51 of $100

### Above your offer line

| Company | Role | Location | Est. year-1 pay (vs offer) | Fit | H-1B | Posted |
|---|---|---|---|---|---|---|
| ByteDance Seed | [Research Scientist, Quantum Chemistry (Global Frontier Tech Program, 2027 start, PhD)](https://joinbytedance.com/search/7632426574734035253) | San Jose, CA | $300K–$450K (+25%) | ●●●●● | sponsors | Open now (2027-start req) |
| Periodic Labs | [Research Scientist / Research Engineer, Materials Data](https://jobs.a16z.com/jobs/periodic-labs/research-scientist-research-engineer-materials-data--ats-mn7ddw8dg6yjkd5er8t6hxq99x8e6krx) | Menlo Park, CA | $280K–$420K (+17%) | ●●●●● | says it sponsors | Sep 1, 2026 |
| ByteDance Seed | [Research Engineer Graduate, Seed Model AI Foundation Model Infrastructure (2027 start, PhD)](https://portfoliojobs.tcv.com/companies/bytedance/jobs/72958826-research-scientist-graduate-seed-ai-foundation-model-infrastructure-2027-start-phd) | Seattle, WA | $300K–$420K (+20%) | ●●● | sponsors | Spring 2026, still taking applications |
| Periodic Labs | [Research Scientist / Research Engineer, Midtraining](https://jobs.a16z.com/jobs/periodic-labs/research-scientist-research-engineer-midtraining--ats-mn783xrp2jmjq4cp3mdjvsjbp98fc7j7) | Menlo Park, CA | $280K–$420K (+17%) | ●●● | says it sponsors | About Sep 30, 2026 |
| NVIDIA | [Research Scientist, Autonomous Systems and Physical AI Research - PhD New College Grad 2026](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Research-Scientist--Autonomous-Systems-and-Physical-AI-Research---PhD-New-College-Grad-2026_JR2027551) **New** | US, CA, Santa Clara | $280K–$420K (+17%) | ●● | sponsors |  |

### Near your offer line

| Company | Role | Location | Est. year-1 pay (vs offer) | Fit | H-1B | Posted |
|---|---|---|---|---|---|---|
| Meta FAIR | [Research Engineer, AI for Chemistry](https://metacareers.com/v2/jobs/3846804338907958) | Seattle, WA | $250K–$340K (-2%) | ●●●●● | sponsors | Live on Meta’s site |
| Google DeepMind | [Research Engineer, Materials Science](https://job-boards.greenhouse.io/deepmind/jobs/7339890) | Mountain View, CA | $230K–$320K (-8%) | ●●●●● | sponsors | May 2026, still taking applications |
| Radical AI | [AI Research Scientist, GNN / Foundation Models](https://jobs.lever.co/RadicalAI/025709b9-d365-4909-aecb-aba5669703da) | New York, NY | $210K–$320K (-12%) | ●●●●● | **ask first** | Live on Radical’s Lever board |
| TikTok | [Machine Learning Scientist Graduate, Trust and Safety (2027 start, PhD)](https://lifeattiktok.com/search/7665985896410368261) | Seattle, WA | $255K–$330K (-3%) | ●●●● | sponsors | About Aug 2026 |
| Snowflake | [AI Research Scientist - New Grad - Agents & Reinforcement Learning](https://jobs.ashbyhq.com/snowflake/1bad12df-f443-426f-9d09-e96fc780d698) **New** | Bellevue, WA | $230K–$330K (-7%) | ●●●● | **ask first** | 2026-09-05 |
| Lila Sciences | [Research Scientist, Computational Condensed Matter Physics](https://jobs.generalcatalyst.com/companies/lila-sciences-2/jobs/87897567-research-scientist-computational-condensed-matter-physics) | Cambridge, MA | $220K–$330K (-8%) | ●●●● | sponsors | Jul 29, 2026 |
| Duolingo | [AI Research Engineer - New PhD Graduate](https://careers.duolingo.com/jobs/8881612002?gh_jid=8881612002) **New** | New York City, NY +1 | $220K–$300K (-13%) | ●●●● | **ask first** | 2026-10-14 |
| NVIDIA | [Research Scientist, Fundamental Generative AI (new college grad)](https://jobs.anitab.org/companies/nvidia/jobs/95232560-research-scientist-fundamental-generative-ai-new-college-grad-2026) **New** | Santa Clara, CA | $270K–$360K (+5%) | ●●● | sponsors | Sep 30, 2026 |
| Meta | [Research Scientist, Machine Learning (PhD), university grad](https://www.metacareers.com/students-and-grads) | Sunnyvale, CA and 7 more | $250K–$320K (-5%) | ●●● | sponsors | Listed on Meta’s Students & Grads page |
| Google | [Software Engineer, AI/Machine Learning, PhD, Early Career (2027 start)](https://www.google.com/about/careers/applications/jobs/results/123087196289671878-software-engineer) | Sunnyvale, Seattle, Kirkland, New York and 6 more | $240K–$310K (-8%) | ●●● | sponsors | Aug 2026 |
| Lila Sciences | [ML Scientist, Foundation Models for Life Sciences](https://job-boards.greenhouse.io/lilasciences/jobs/4222051009) **New** | San Francisco, CA USA | $200K–$330K (-12%) | ●●● | sponsors | 2026-04-28 |

### Not screened yet
These passed the keyword filters but weren't screened (AI off or the monthly budget is used up).

| Company | Role | Location | Found via |
|---|---|---|---|
| Lila Sciences | [Machine Learning Engineer, Physical Sciences AI](https://job-boards.greenhouse.io/lilasciences/jobs/4116513009) | Cambridge, MA USA; San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Machine Learning Scientist I / II, Protein Design](https://job-boards.greenhouse.io/lilasciences/jobs/4396760009) | San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Microfabrication Scientist I/II](https://job-boards.greenhouse.io/lilasciences/jobs/4400746009) | Cambridge, MA USA | Greenhouse (lilasciences) |
| Lila Sciences | [ML Scientist I/II, AI for Protein Engineering](https://job-boards.greenhouse.io/lilasciences/jobs/4392245009) | San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [ML Scientist, Nucleic Acid Design](https://job-boards.greenhouse.io/lilasciences/jobs/4324969009) | San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Platform Scientist I/II, Functional Materials Instrumentation](https://job-boards.greenhouse.io/lilasciences/jobs/4423504009) | Cambridge, MA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Research Engineer, Scientific Data & Workflow Infrastructure](https://job-boards.greenhouse.io/lilasciences/jobs/4286513009) | Cambridge, MA USA; San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Scientist I/II, Characterization and Composition Analysis](https://job-boards.greenhouse.io/lilasciences/jobs/4384606009) | Cambridge, MA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Scientist I/II, X-ray Diffraction Characterization](https://job-boards.greenhouse.io/lilasciences/jobs/4378385009) | Cambridge, MA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Scientist II/Senior Characterization Scientist,  Condensed Matter](https://job-boards.greenhouse.io/lilasciences/jobs/4246305009) | Cambridge, MA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Scientist II/ Senior Scientist, BioML](https://job-boards.greenhouse.io/lilasciences/jobs/4395729009) | San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Scientist II/Senior Scientist, Computational Biophysics](https://job-boards.greenhouse.io/lilasciences/jobs/4340155009) | Cambridge, MA USA; London, UK; San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Scientist II / Senior Scientist, Electron Diffraction Characterization](https://job-boards.greenhouse.io/lilasciences/jobs/4378383009) | Cambridge, MA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Scientist II/Senior Scientist, Solid-State Materials](https://job-boards.greenhouse.io/lilasciences/jobs/4271809009) | Cambridge, MA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Scientist II, Silicon Photonics](https://job-boards.greenhouse.io/lilasciences/jobs/4383465009) | Cambridge, MA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Senior Data Engineer, Bioinformatics, Cheminformatics, Materials](https://job-boards.greenhouse.io/lilasciences/jobs/4377096009) | San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Senior / Engineer II, AI Lab Research Engineer](https://job-boards.greenhouse.io/lilasciences/jobs/4029507009) | Cambridge, MA USA; San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Senior ML Scientist, AI for Protein Engineering](https://job-boards.greenhouse.io/lilasciences/jobs/4392247009) | San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Senior ML Scientist, Biological Systems](https://job-boards.greenhouse.io/lilasciences/jobs/4395725009) | San Francisco, CA USA | Greenhouse (lilasciences) |
| Lila Sciences | [Software Engineer, AI Platform](https://job-boards.greenhouse.io/lilasciences/jobs/4031328009) | Cambridge, MA USA | Greenhouse (lilasciences) |
| Anthropic | [Applied AI Engineer, Beneficial Deployments (Life Sciences)](https://job-boards.greenhouse.io/anthropic/jobs/5437172008) | San Francisco, CA \| New York City, NY | Greenhouse (anthropic) |
| Anthropic | [Applied AI Engineer, Startups (Life Sciences)](https://job-boards.greenhouse.io/anthropic/jobs/5454650008) | San Francisco, CA \| New York City, NY | Greenhouse (anthropic) |
| Anthropic | [Applied AI Engineer, Startups (Physical AI)](https://job-boards.greenhouse.io/anthropic/jobs/5454673008) | San Francisco, CA | Greenhouse (anthropic) |
| Anthropic | [Research Engineer, Life Sciences](https://job-boards.greenhouse.io/anthropic/jobs/5265365008) | San Francisco, CA | Greenhouse (anthropic) |
| Anthropic | [Research Scientist, Life Sciences](https://job-boards.greenhouse.io/anthropic/jobs/5239733008) | San Francisco, CA | Greenhouse (anthropic) |
| Anthropic | [Research Scientist, Life Sciences (Chemistry)](https://job-boards.greenhouse.io/anthropic/jobs/5387564008) | San Francisco, CA | Greenhouse (anthropic) |
| Anthropic | [Research Scientist, Life Sciences (Computational)](https://job-boards.greenhouse.io/anthropic/jobs/5357739008) | San Francisco, CA | Greenhouse (anthropic) |
| Anthropic | [Research Scientist, Life Sciences (Computational Chemistry)](https://job-boards.greenhouse.io/anthropic/jobs/5461863008) | San Francisco, CA | Greenhouse (anthropic) |
| Anthropic | [Research Scientist, Life Sciences (Experimental Biology)](https://job-boards.greenhouse.io/anthropic/jobs/5285250008) | San Francisco, CA | Greenhouse (anthropic) |
| Waymo | [Machine Learning Engineer Perception LLM/VLM (PhD, New Grad)](https://careers.withwaymo.com/jobs?gh_jid=7488508) | Mountain View, CA USA;  San Francisco, CA USA; | Greenhouse (waymo) |
| Pinterest | [PhD University Grad Machine Learning Engineer (USA) ](https://www.pinterestcareers.com/jobs/?gh_jid=8140363) | San Francisco, CA, US; Palo Alto, CA, US; Seattle, WA, US | Greenhouse (pinterest) |
| Periodic Labs | [Research Scientist, Materials Characterization](https://jobs.ashbyhq.com/periodic-labs/e90c6050-e544-4a2b-ae30-7c41e5eb4f16) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [Research Scientist, Thin Films](https://jobs.ashbyhq.com/periodic-labs/7f36ea6e-3e67-40df-82c6-5540859a2ac5) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [Research Scientist, Condensed Matter Theory](https://jobs.ashbyhq.com/periodic-labs/01d2c4b2-d848-4426-ad5f-7f1052db3a63) | Menlo Park, CA; Montreal, Canada | Ashby (periodic-labs) |
| Periodic Labs | [Computational Scientist, Structural & Thermal](https://jobs.ashbyhq.com/periodic-labs/0d96f03c-8f08-44ae-a7cb-8c4bbc80976c) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [Computational Scientist, Differentiable Physics](https://jobs.ashbyhq.com/periodic-labs/ccdb34b1-b67c-4dbf-8acb-fe786f2ab38b) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [Research Scientist, Data](https://jobs.ashbyhq.com/periodic-labs/d039ec58-a97d-47e1-b1e5-db003d772f77) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [ML Systems Engineer](https://jobs.ashbyhq.com/periodic-labs/d9180228-c113-4d5f-a4d6-793994e50b7c) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [Forward Deployed Engineer, Physics & Simulation](https://jobs.ashbyhq.com/periodic-labs/1bab7e17-20fd-4b82-a152-98b96375a626) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [Research Engineer, Semiconductor](https://jobs.ashbyhq.com/periodic-labs/95d290bd-6420-42e8-82a0-f0230722c19c) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [Research Engineer, Lab Automation](https://jobs.ashbyhq.com/periodic-labs/a16ee7dd-021d-4f37-823d-7a8b520c6d8a) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [Research Associate, Thin Films ](https://jobs.ashbyhq.com/periodic-labs/98837207-657f-4116-a112-d8dc6184b49e) | Menlo Park, CA | Ashby (periodic-labs) |
| Periodic Labs | [Research Scientist, Scaling RL](https://jobs.ashbyhq.com/periodic-labs/20b122c9-b8ec-4fb0-aaf4-9b45902affe0) | Menlo Park, CA; Montreal, Canada | Ashby (periodic-labs) |
| OpenAI | [Applied Data Science & Insights Leader - GTM Intelligence Solutions and Technical Success](https://jobs.ashbyhq.com/openai/93f558c1-c944-4ddc-b102-addc9446de3d) | San Francisco | Ashby (openai) |
| NVIDIA | [Research Scientist, Robotics Research -  PhD New College Grad 2026](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-WA-Seattle/Research-Scientist--Robotics-Research----PhD-New-College-Grad-2026_JR2011473) | US, WA, Seattle | Workday (NVIDIA) |
| NVIDIA | [Research Scientist, ML Systems - PhD New College Grad 2026](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Research-Scientist--ML-Systems---New-College-Grad-2026_JR2010161) | 4 Locations | Workday (NVIDIA) |
| NVIDIA | [Research Scientist, Electronic Design Automation - New College Grad 2026](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Research-Scientist--Electronic-Design-Automation---New-College-Grad-2026_JR2012979) | US, CA, Santa Clara | Workday (NVIDIA) |
| NVIDIA | [Research Scientist, Physical AI - Foundation Models - PhD New College Grad 2026](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Research-Scientist--Generative-AI-for-Physical-AI---PhD-New-College-Grad-2026_JR2016032) | US, CA, Santa Clara | Workday (NVIDIA) |
| NVIDIA | [Research Scientist, AI for Graphics and Gaming - New College Grad 2026](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Research-Scientist--AI-for-Graphics-and-Gaming---New-College-Grad-2026_JR2011267) | US, CA, Santa Clara | Workday (NVIDIA) |
| NVIDIA | [High-Performance LLM Training Engineer - New College Grad 2027](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/High-Performance-LLM-Training-Engineer---New-College-Grad-2027_JR2028237) | US, CA, Santa Clara | Workday (NVIDIA) |
| NVIDIA | [NVIDIA 2027 New College Graduate: Deep Learning and High-Performance Computing Engineering - China](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/China-Shanghai/NVIDIA-2027-New-College-Graduate--Deep-Learning-and-High-Performance-Computing-Engineering---China_JR2024107) | 3 Locations | Workday (NVIDIA) |
| Google | [Software Engineer, AI/Machine Learning,  PhD, Early Career, 2027 Start](https://www.google.com/about/careers/applications/jobs/results/123087196289671878-software-engineer-aimachine-learning-phd-early-career-2027-start) | United States | Google careers search |
| Google | [Student Researcher, PhD, Winter-Summer 2027](https://www.google.com/about/careers/applications/jobs/results/134313315235963590-student-researcher-phd-wintersummer-2027) | United States | Google careers search |
| Google | [Student Researcher, BS/MS, Winter-Summer 2027](https://www.google.com/about/careers/applications/jobs/results/131518356678156998-student-researcher-bsms-wintersummer-2027) | United States | Google careers search |
| Google | [Software Engineer, Systems Research, PhD, Early Career](https://www.google.com/about/careers/applications/jobs/results/76908711519363782-software-engineer-systems-research-phd-early-career) | United States | Google careers search |
| Google | [Vice President, Product Management, Core AI Foundations](https://www.google.com/about/careers/applications/jobs/results/113245419391066822-vice-president-product-management-core-ai-foundations) | United States | Google careers search |
| Google | [Student Researcher, PhD, Fall 2026](https://www.google.com/about/careers/applications/jobs/results/83535564552184518-student-researcher-phd-fall-2026) | United States | Google careers search |
| Google | [Research Scientist, Science AI, Google Research](https://www.google.com/about/careers/applications/jobs/results/91797366327124678-research-scientist-science-ai-google-research) | United States | Google careers search |
| Pacific Northwest National Laboratory | [Materials Scientist 3 - Machine Learning for Materials Discovery](https://careers.conbio.org/job/1349634/materials-scientist-3-machine-learning-for-materials-discovery/?TrackID=9) | Richland, WA | Claude web search: machine learning interatomic potential research scientist jo |
| Applied Materials | [2026 New College Grad - Computational Chemist / Materials Scientist (Machine Learning - Reactive MLIPs) - Doctorate Degree](https://getmereferred.com/us/job-listing/2026-new-college-grad-computational-chemist-materials-scientist-machine-learning-reactive-mlips-doctorate-degree-applied-materials-uscasantaclara-2-to-5-years-experience-fabedcc4-cf7e-4580-bdc5-313ec5c23dc4) | Santa Clara, CA | Claude web search: crystal structure prediction machine learning job |

### Closed recently

- Meta FAIR, AI Research Scientist, FAIR Chemistry (New York) (2026-10-25)
- Lila Sciences, ML Research Scientist I/II, Multimodal Data Extraction (2026-10-25)

<details><summary>Screened out (30)</summary>

- [Solace, Associate Data Scientist - College Grad 2027](https://jobs.ashbyhq.com/solace/77ca492c-4142-4931-beeb-e85d9d0ac443): Estimated $150K–$210K, below your line.
- [Hewlett Packard Enterprise, AI and Machine Learning Engineering Graduate](https://hpe.wd5.myworkdayjobs.com/en-US/jobsathpe/job/Durham-North-Carolina-United-States-of-America/AI-and-Machine-Learning-Engineering-Graduate_1213475): Estimated $150K–$200K, below your line.
- [Vertex, Computational Chemistry Research Scientist - Computational Drug Design](https://vrtx.wd501.myworkdayjobs.com/en-US/vertex_careers/job/Boston-MA/Computational-Chemistry-Research-Scientist---Computational-Drug-Design--_REQ-30307): Estimated $150K–$210K, below your line.
- [Brain Co., AI Security Engineer - 2027 New Grad](https://jobs.ashbyhq.com/brainco/32c4e1c9-ccd1-4dca-a68b-31ad1b868030): Estimated $200K–$280K, below your line.
- [Brain Co., AI/ML Engineer - 2027 New Grad](https://jobs.ashbyhq.com/brainco/4c9f8f00-0caf-4927-80da-02a3e53cb0cd): Estimated $200K–$280K, below your line.
- [Brain Co., AI Platform Engineer - 2027 New Grad](https://jobs.ashbyhq.com/brainco/4d39773e-f4c3-49c1-87c3-812a9804fb15): Estimated $200K–$280K, below your line.
- [General Motors, Entry-Level - AI and Machine Learning for Automation Researcher](https://generalmotors.wd5.myworkdayjobs.com/en-US/careers_gm/job/Warren-Michigan-United-States-of-America/Entry-Level---AI-and-Machine-Learning-for-Automation-Researcher_JR-202621769): Estimated $160K–$220K, below your line.
- [General Motors, Entry-Level - AI/Machine Learning Researcher](https://generalmotors.wd5.myworkdayjobs.com/en-US/careers_gm/job/Mountain-View-California-United-States-of-America/Entry-Level---AI-Machine-Learning-Researcher_JR-202621753): Estimated $150K–$210K, below your line.
- [Quora, Machine Learning Engineer - New Grad - Quora - Remote](https://jobs.ashbyhq.com/quora/40659fe7-545e-4b08-b6d7-2567b18b9ed3): Estimated $170K–$240K, below your line.
- [GlobalFoundries, AI/ML Systems Engineer - 2027 New College Graduate](https://globalfoundries.wd1.myworkdayjobs.com/en-US/external/job/USA---Texas---Richardson/AI-ML-Systems-Engineer--2027-New-College-Graduate-_JR-2604163): Estimated $145K–$185K, below your line.
- [Brain Co., AI Product Engineer - 2027 New Grad](https://jobs.ashbyhq.com/brainco/c1788246-305f-4e4f-8b03-8444c76a466e): Estimated $200K–$280K, below your line.
- [Applied Underwriters, AI Engineer - Data Science](https://jobs.lever.co/auw/6f317af6-4362-4dea-accc-e86ec1d102a5): Estimated $150K–$210K, below your line.
- [Applied Materials, AI Research Scientist (Generative Models for Scientific Discovery)](https://jobs.appliedmaterials.com/job/santa-clara/ai-research-scientist-generative-models-for-scientific-discovery/95/95703498768): Estimated $190K–$250K, below your line.
- [Lila Sciences, Research Scientist I/II, In Silico Materials Discovery](https://www.indeed.com/q-machine-learning-research-scientist-phd-jobs.html): Estimated $200K–$270K, below your line.
- [Applied Materials, 2026 New College Grad - Computational Chemist / Materials Scientist (Machine Learning - Reactive MLIPs)](https://www.indeed.com/q-research-scientist-machine-learning-applied-math-jobs.html): Estimated $150K–$190K, below your line.
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
