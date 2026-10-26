# Plan a 30-minute research talk on an ML-accelerated crystal structure prediction engine

You have a 30-minute slot plus 15 minutes of questions at a research seminar with about 40 people: half ML researchers who have never relaxed a crystal structure, half chemists and materials scientists who are skeptical of ML. The work is an evolutionary crystal structure prediction engine that relaxes and ranks candidates with a fine-tuned ML interatomic potential and verifies the shortlist with DFT. Your numbers: relaxations are 20,000 times cheaper than DFT; the engine recovers the known ground state in 46 of 50 benchmark systems versus 36 of 50 for a DFT-only search at the same compute; and it found 14 new structures within 25 meV/atom of the hull across 6 chemical systems, 9 of them confirmed by DFT so far. Plan the talk.

Spend 30 minutes. Write the plan as if you were walking a mentor through it the day before the talk, in this order:

1. **The claim and the first 2 minutes.** The one sentence the talk defends, the headline number, and why each half of the room should care. No history of the field.
2. **Structure and time budget.** A slide-by-slide outline with minutes per section: problem, method, key result, ablations and limitations, what to reuse. Say what you cut to fit.
3. **The one key figure.** What it plots, the baseline and error bars on it, how both audiences read it without help, and what you say in the 90 seconds it is on screen.
4. **What the audience can reuse.** One concrete thing for ML researchers (a search or evaluation idea) and one for chemists (how to run or validate it), plus the limitations slide that names where the engine fails.
5. **Hostile questions.** The five questions you least want (for example MLIP errors near the hull, benchmark leakage from the potential's training data, the choice of functional, cherry-picked systems, how this differs from existing CSP tools), a two-sentence honest answer for each, and your rule for a question you cannot answer.

Then score yourself with the rubric, one point per item, or paste your answer into a Claude session and ask for a rubric-based score with specific feedback.

## What the interviewer listens for

Whether the talk has one claim and every slide serves it. Whether the key figure carries a baseline and uncertainty, and whether you would state the limitations before the audience does. Whether your answers to attacks concede what is true and point to data instead of defending.
