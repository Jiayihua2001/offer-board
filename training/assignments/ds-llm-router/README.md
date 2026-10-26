# Design a cost-aware LLM routing system

You run a classification service that labels product listings against a policy, at roughly 200K cases a day. A frontier model gets 96% of cases right but costs 40 times more per call than a small model that gets 88% right. Design a system that keeps overall accuracy at or above 95% while cutting cost as much as possible.

Spend 30 minutes. Write your answer as if you were talking through it with an interviewer, in this order:

1. **Requirements and constraints.** Accuracy, cost, latency, volume, and what a mistake costs. Ask what you'd want to clarify.
2. **Routing policy.** How a request is assigned to a model. Cascade with a confidence gate, a learned router trained on where the models disagree, or a hybrid. Say how you'd pick the thresholds.
3. **Quality estimation.** What signals tell you the small model is likely right (token log-probs, a trained quality estimator, agreement between two cheap samples). How you calibrate those signals so a threshold means what it says.
4. **Evaluation.** Offline replay on a labeled set, the cost-versus-accuracy frontier, and an online experiment with guardrails.
5. **Operations.** Drift, model version changes, monitoring, and how the human-label loop feeds back.

Then score yourself with the rubric, one point per item, or paste your answer into a Claude session and ask for a rubric-based score with specific feedback.

## What the interviewer listens for

Whether you reason from the requirement to the design rather than reciting components. Whether you know how to set a threshold from data and quantify what you'd lose. Whether you have an answer for "the frontier model got updated last night; what happens to your router?"
