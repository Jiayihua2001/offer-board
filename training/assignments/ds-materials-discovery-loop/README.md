# Design an active-learning loop for battery cathode discovery

You have 40 weeks and a fixed DFT allocation of 20,000 structure relaxations (about 500 per week, roughly 5 million core-hours in total) to find new Li-ion cathode materials. The targets: energy above the convex hull at or below 30 meV/atom, average voltage between 3.5 and 4.5 V versus Li/Li+, theoretical capacity above 200 mAh/g, and volume change on delithiation under 8%. The candidate pool is about 2 million structures from prototype substitution plus an evolutionary generator. You have a universal ML interatomic potential (MACE-MP or UMA class) fine-tuned on 50,000 in-domain calculations that relaxes a structure 10,000 times faster than DFT, but its errors grow with novelty. Random screening of this pool lands about 1 in 200 DFT relaxations on or near the hull. Design the loop that spends the DFT budget best.

Spend 30 minutes. Write your answer as if you were talking through it with an interviewer, in this order:

1. **Requirements and objective.** Define a stable new cathode in numbers: hull distance, the voltage window from lithiated and delithiated energies, capacity from composition, volume change. State the budget per round, the baseline hit rate, and what success looks like at week 40. Ask what you'd want to clarify.
2. **Candidate generation and surrogate.** How candidates are produced, how the MLIP relaxes and ranks them, how you estimate voltage and uncertainty, how you deduplicate, and what one round costs in MLIP compute.
3. **Acquisition and batching.** How you choose the 500 structures per week: exploitation on predicted hull distance, exploration on uncertainty, diversity across chemistries and prototypes, and a reserved random or uncertainty-only slice. Say how the hull is rebuilt when new phases arrive.
4. **Validation and retraining.** DFT settings consistent with the reference database, how each round's results flow back into the MLIP (fine-tune or retrain, and when), how you detect that the surrogate has drifted off the current search region, and when you stop.
5. **Evaluation and failure modes.** Discovery rate versus the baseline at equal budget, uncertainty calibration, what you report at the end, and the handoff to synthesis. Name what goes wrong when an evolutionary search meets an imperfect potential.

Then score yourself with the rubric, one point per item, or paste your answer into a Claude session and ask for a rubric-based score with specific feedback.

## What the interviewer listens for

Whether you treat DFT relaxations as the scarce resource and judge every choice by hits per relaxation. Whether you know that the surrogate's top-ranked candidates are exactly where its errors are largest, and build the acquisition and the batch to survive that. Whether you can say what the hull is, why it moves, and how you would know the loop has stopped paying off.
