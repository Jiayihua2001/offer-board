import numpy as np

from solution import cosine_topk, mmr, normalize


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def mmr_reference(query, docs, k, lam):
    """Deliberately naive greedy MMR with explicit loops."""
    n = len(docs)
    rel = np.array([cosine(query, docs[i]) for i in range(n)])
    selected = [int(np.argmax(rel))]
    while len(selected) < k:
        best, best_score = None, -np.inf
        for i in range(n):
            if i in selected:
                continue
            redundancy = max(cosine(docs[i], docs[s]) for s in selected)
            score = lam * rel[i] - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
    return selected


def test_normalize_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3)) * 10
    X[2] = 0.0
    Xn = normalize(X)
    assert Xn.shape == X.shape
    assert np.allclose(np.linalg.norm(Xn[[0, 1, 3, 4]], axis=1), 1.0)
    assert np.all(Xn[2] == 0.0) and np.all(np.isfinite(Xn)), "a zero row must stay zero, not become nan"
    assert np.allclose(Xn[0], X[0] / np.linalg.norm(X[0]))


def test_topk_matches_full_sort():
    rng = np.random.default_rng(1)
    docs = rng.normal(size=(2000, 32))
    q = rng.normal(size=32)
    k = 10
    idx, scores = cosine_topk(q, docs, k)
    idx, scores = np.asarray(idx), np.asarray(scores)
    sims = (docs / np.linalg.norm(docs, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
    ref = np.argsort(-sims)[:k]
    assert idx.shape == (k,) and scores.shape == (k,)
    assert np.array_equal(idx, ref), f"got {idx}, expected {ref}"
    assert np.allclose(scores, sims[ref], atol=1e-12)
    assert np.all(np.diff(scores) <= 0), "scores must be sorted from most to least similar"


def test_topk_is_invariant_to_document_scale():
    rng = np.random.default_rng(2)
    docs = rng.normal(size=(300, 8))
    q = rng.normal(size=8)
    idx, scores = cosine_topk(q, docs, 5)
    scaled = docs * rng.uniform(0.01, 100.0, size=(300, 1))
    idx2, scores2 = cosine_topk(q, scaled, 5)
    assert np.array_equal(np.asarray(idx), np.asarray(idx2))
    assert np.allclose(scores, scores2, atol=1e-10)
    idx3, _ = cosine_topk(q * 1000.0, docs, 5)
    assert np.array_equal(np.asarray(idx), np.asarray(idx3))


def test_topk_edge_sizes():
    rng = np.random.default_rng(3)
    docs = rng.normal(size=(7, 4))
    q = rng.normal(size=4)
    sims = (docs / np.linalg.norm(docs, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
    idx1, s1 = cosine_topk(q, docs, 1)
    assert np.asarray(idx1).tolist() == [int(np.argmax(sims))] and np.isclose(s1[0], sims.max())
    idx_all, s_all = cosine_topk(q, docs, 7)
    assert np.array_equal(np.asarray(idx_all), np.argsort(-sims))
    assert np.allclose(s_all, np.sort(sims)[::-1])


def test_mmr_with_lambda_one_is_topk():
    rng = np.random.default_rng(4)
    docs = rng.normal(size=(100, 16))
    q = rng.normal(size=16)
    idx, _ = cosine_topk(q, docs, 6)
    assert list(mmr(q, docs, 6, lam=1.0)) == np.asarray(idx).tolist()


def test_mmr_suppresses_duplicates():
    q = np.array([1.0, 0.0])
    docs = np.array([[1.0, 0.2], [1.0, 0.2], [1.0, -0.4], [0.0, 1.0]])
    idx, _ = cosine_topk(q, docs, 2)
    assert set(np.asarray(idx).tolist()) == {0, 1}, "plain top-k returns both duplicates"
    assert list(mmr(q, docs, 2, lam=0.5)) == [0, 2]
    assert list(mmr(q, docs, 2, lam=1.0)) == [0, 1]
    assert list(mmr(q, docs, 3, lam=0.5)) == [0, 2, 1], "with nothing relevant left, the duplicate beats an irrelevant doc"
    assert list(mmr(q, docs, 3, lam=0.2)) == [0, 3, 2], "a small lambda prefers the orthogonal doc over a near-duplicate"


def test_mmr_matches_greedy_reference_and_is_well_formed():
    rng = np.random.default_rng(5)
    docs = rng.normal(size=(300, 12))
    q = rng.normal(size=12)
    for lam in [0.3, 0.6, 0.9]:
        out = list(mmr(q, docs, 8, lam=lam))
        assert len(out) == 8 and len(set(out)) == 8, "k distinct indices"
        assert out[0] == int(cosine_topk(q, docs, 1)[0][0]), "the first pick is the most relevant document"
        assert out == mmr_reference(q, docs, 8, lam), f"lam={lam}: {out} vs reference"
