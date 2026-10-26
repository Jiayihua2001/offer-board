# Design the evaluation pipeline for an LLM classifier replacing human reviewers

A trust-and-safety team reviews about 50,000 content items a day against a 12-category policy with 40 contract reviewers at roughly $0.60 per decision. On a double-labeled sample, human-human agreement is Cohen's kappa 0.71 overall and 0.55 on the three hardest categories; violation prevalence ranges from 6% down to 0.4% per category. You are replacing most of that review with an LLM classifier that auto-decides and escalates the rest to humans. Leadership will approve launch only if decision quality is at least that of a median reviewer in every category and cost per decision falls by 60%. Design the evaluation pipeline: the gold set, how you measure and use label disagreement, how you sample for audits, the release gate, and what runs after launch.

Spend 30 minutes. Write your answer as if you were talking through it with an interviewer, in this order:

1. **Requirements.** What "as good as a median reviewer" means operationally, the cost of a false positive and a false negative per category, volume and latency, and which decisions stay with humans regardless. Ask what you'd want to clarify.
2. **Gold set.** How you sample items (stratified by category and by model confidence, enriched for rare violations), how many annotations per item and who adjudicates, how many items you need for the confidence intervals you want in the 0.4% category, and how you keep the set uncontaminated.
3. **Disagreement analysis.** Model-human and human-human agreement on the same items, confusion matrices per category, adjudicated attribution of who was right, and what you do with categories where the policy itself is ambiguous.
4. **Release gate.** Shadow mode, the statistical test and margin for non-inferiority, per-category floors, how the decision threshold and calibration are fixed before the test, and the canary rollout and rollback rule.
5. **Operations.** The ongoing audit sample and its size, weekly dashboards, drift alarms, what happens when the prompt or model version changes, and how audited labels feed back into training.

Then score yourself with the rubric, one point per item, or paste your answer into a Claude session and ask for a rubric-based score with specific feedback.

## What the interviewer listens for

Whether you know that human agreement is the ceiling and measure the model against it on the same items, per category. Whether your sample sizes come from the rarest category and your gold set cannot leak into prompts or training. Whether the release gate is a test with a stated margin rather than a number that looked good, and whether you have a plan for the week the prompt or the model version changes.
