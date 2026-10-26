import numpy as np


def normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(norms, eps)


def cosine_topk(query: np.ndarray, docs: np.ndarray, k: int):
    sims = normalize(docs) @ normalize(query)                 # (n,)
    cand = np.argpartition(-sims, k - 1)[:k]                  # O(n): the top k, unordered
    order = cand[np.argsort(-sims[cand], kind="stable")]      # O(k log k): rank just those
    return order, sims[order]


def mmr(query: np.ndarray, docs: np.ndarray, k: int, lam: float = 0.5) -> list:
    docs_n = normalize(docs)
    rel = docs_n @ normalize(query)                           # relevance of every document
    selected = [int(np.argmax(rel))]
    redundancy = docs_n @ docs_n[selected[0]]                 # max similarity to the selected set, so far one doc
    for _ in range(1, k):
        score = lam * rel - (1.0 - lam) * redundancy
        score[selected] = -np.inf
        i = int(np.argmax(score))
        selected.append(i)
        redundancy = np.maximum(redundancy, docs_n @ docs_n[i])
    return selected
