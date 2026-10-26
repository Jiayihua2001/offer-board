# Design retrieval-augmented QA over 2 million materials papers

About 300 materials scientists want to ask questions of a corpus of 2 million full-text papers (average 15,000 tokens each, with tables, figures and supplementary files), growing by 2,000 papers a week. Typical questions: which synthesis routes give phase-pure LiNi0.8Mn0.1Co0.1O2 with low cation mixing, what ionic conductivities have been reported for a given garnet composition, or which papers contradict a claimed band gap. Users expect an answer with citations in under 10 seconds, and a confident wrong answer with real-looking citations is the failure that would kill adoption. You have 4 inference GPUs, an embedding budget of one pass over the corpus per quarter, and access to a frontier model API. Design the system and its evaluation.

Spend 30 minutes. Write your answer as if you were talking through it with an interviewer, in this order:

1. **Requirements.** The query types (fact lookup, recipe extraction, numeric property lookup, cross-paper comparison), latency, freshness, the citation requirement, and what an acceptable failure looks like (abstain versus wrong). Ask what you'd want to clarify.
2. **Ingestion and indexing.** PDF parsing, chunking of sections, tables, equations and captions, metadata, the embedding model and whether to adapt it, the chunk count and index size, hybrid sparse plus dense retrieval, and deduplication of preprints and versions.
3. **Retrieval and generation.** Query rewriting and chemistry-aware normalization, candidate retrieval and reranking, how many chunks go into the prompt and how they are cited, abstention when the evidence is weak, and structured extraction for numeric questions.
4. **Evaluation.** How you build a test set with expert answers and gold passages, retrieval metrics, answer faithfulness and correctness, citation precision, how an LLM judge is validated against experts, and the latency and cost per query.
5. **Operations.** Weekly ingestion and index updates, capturing user feedback and corrections, monitoring retrieval drift, defending against instructions embedded in paper text, and the cost of running it.

Then score yourself with the rubric, one point per item, or paste your answer into a Claude session and ask for a rubric-based score with specific feedback.

## What the interviewer listens for

Whether you do the corpus arithmetic before choosing an index, and whether your retrieval handles chemical formulas, units and synonyms rather than treating papers as generic text. Whether your evaluation separates retrieval failure from generation failure and measures citation faithfulness. Whether you can say when the system should say it does not know.
