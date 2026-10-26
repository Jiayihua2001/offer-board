# Embedding retrieval: cosine top-k and MMR re-ranking

The retrieval half of every RAG or recommendation system is three small NumPy functions, and the interview question is usually "write the top-k, now make the results less redundant". It tests whether you normalize once instead of per query, whether you know `argpartition`, and whether you can state maximal marginal relevance as a greedy objective rather than a vibe.

- `normalize(X, eps=1e-12)`: rows of `X` (shape `(n, d)`) scaled to unit length; an all-zero row stays zero.
- `cosine_topk(query, docs, k) -> (indices, scores)`: `query` has shape `(d,)`, `docs` has shape `(n, d)`. Return the `k` indices with the largest cosine similarity, sorted from most to least similar, and their scores. Use `argpartition` to find the candidates in `O(n)`, then sort only the `k`.
- `mmr(query, docs, k, lam=0.5) -> list of k indices`: maximal marginal relevance. Start with the most similar document; then repeatedly add the unselected document `i` that maximizes `lam * cos(query, d_i) - (1 - lam) * max over selected s of cos(d_i, d_s)`. Return the indices in the order they were selected. `lam = 1` reduces to the top-k ranking; smaller `lam` trades relevance for diversity.

## Example

```
q    = np.array([1.0, 0.0])
docs = np.array([[1.0, 0.2], [1.0, 0.2], [1.0, -0.4], [0.0, 1.0]])   # docs 0 and 1 are duplicates
cosine_topk(q, docs, 2)   -> indices {0, 1} (either order: identical scores 0.981), scores [0.981, 0.981]
mmr(q, docs, 2, lam=0.5)  -> [0, 2]   (doc 1 is redundant with doc 0; doc 2 is nearly as relevant and different)
mmr(q, docs, 2, lam=1.0)  -> [0, 1]
```

## Constraints

- No loops over documents: similarities are one matrix-vector product; the only loop in `mmr` is over the `k` picks, each `O(n d)` with a running maximum.
- `cosine_topk` must agree exactly with a full `argsort` on random data, and must be invariant to scaling any document by a positive factor.
- `mmr` must return `k` distinct indices, the first being the top-1 document.

## What the interviewer listens for

- Why cosine instead of dot product or Euclidean distance (length-invariance; for normalized vectors all three rank identically) and why you pre-normalize the corpus once (the dominant cost is `O(n d)` per query, so never redo the norms).
- `argpartition` versus `argsort`: `O(n)` versus `O(n log n)`, and when you would move to an ANN index (HNSW, IVF-PQ) because even `O(n d)` is too slow at 100 M documents.
- MMR as a greedy approximation to a relevance-minus-redundancy objective: `lam` is the knob, the greedy choice is `O(k n)` similarity lookups, and duplicates get suppressed because their redundancy term is 1.
- Follow-up: how you would evaluate retrieval (recall@k, MRR, nDCG against labeled relevance), and where a cross-encoder re-ranker fits relative to MMR in a RAG pipeline.
