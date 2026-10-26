# Design retrieval and ranking for a short-video feed

A short-video app has 50 million daily active users and a catalog of 500 million videos with 5 million new uploads a day. Each feed request returns 8 videos; users make about 40 requests a day, so the service handles about 2 billion requests a day, roughly 25,000 requests per second on average and 3 times that at peak. The recommendation service has a 150 ms p99 latency budget end to end. The product goal is long-term engagement, measured by 30-day retention and daily watch time, with new uploads needing exposure within an hour and policy-violating content never shown. Design the two-stage system: candidate generation, ranking, features, training data, serving, and metrics.

Spend 30 minutes. Write your answer as if you were talking through it with an interviewer, in this order:

1. **Requirements and objective.** What the feed optimizes and what proxies you can train on, the constraints (freshness, diversity, policy filters), the scale numbers, and the latency budget per stage. Ask what you'd want to clarify.
2. **Candidate generation.** The two-tower model (user tower, item tower, similarity), how it is trained with in-batch negatives and why that needs a correction, the approximate nearest-neighbour index, and the other candidate sources (co-visitation, followed creators, a fresh-upload pool). How many candidates leave this stage.
3. **Ranking.** A multi-task ranker: the labels it predicts, the features (user history sequence, item, context, crosses), the architecture, how heads are combined into one score, calibration, and re-ranking for diversity and business rules.
4. **Training data and pipeline.** Impression logs and delayed labels, negatives for each stage, position bias, feedback loops, how often each model retrains, and the size of the embedding tables.
5. **Serving and metrics.** Feature store, precomputed item embeddings, request-time user embedding, caching, offline metrics per stage, the online A/B design with guardrails, and cold start for new users and new videos.

Then score yourself with the rubric, one point per item, or paste your answer into a Claude session and ask for a rubric-based score with specific feedback.

## What the interviewer listens for

Whether you split the latency budget and size the models to it. Whether you know how a two-tower model is trained and why in-batch negatives need a correction. Whether you can name where the logged training data is biased by the system that produced it and what you do about it, and whether your offline metrics would predict the online ones.
