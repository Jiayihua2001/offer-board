import numpy as np

from solution import gcn_forward, gcn_layer, graph_readout, normalize_adjacency


def make_params(d_in, d_hidden, d2, d_out, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(size=(d_in, d_hidden)) / np.sqrt(d_in), "b1": rng.normal(size=d_hidden) * 0.1,
        "W2": rng.normal(size=(d_hidden, d2)) / np.sqrt(d_hidden), "b2": rng.normal(size=d2) * 0.1,
        "W_out": rng.normal(size=(d2, d_out)) / np.sqrt(d2), "b_out": rng.normal(size=d_out) * 0.1,
    }


def random_graph(n, p, seed):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p).astype(float)
    A = np.triu(A, 1)
    return A + A.T


def path_graph(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


def test_normalized_adjacency_values():
    A_hat = normalize_adjacency(path_graph(3))
    s = 1 / np.sqrt(6.0)
    assert np.allclose(A_hat, [[0.5, s, 0.0], [s, 1 / 3, s], [0.0, s, 0.5]])
    assert np.allclose(A_hat, A_hat.T)
    cycle = np.roll(np.eye(4), 1, axis=1) + np.roll(np.eye(4), -1, axis=1)     # 2-regular -> (A + I) / 3
    assert np.allclose(normalize_adjacency(cycle), (cycle + np.eye(4)) / 3)


def test_isolated_nodes_and_spectrum():
    A = path_graph(3)
    A = np.pad(A, ((0, 1), (0, 1)))                   # node 3 is isolated
    A_hat = normalize_adjacency(A)
    assert np.all(np.isfinite(A_hat))
    assert np.isclose(A_hat[3, 3], 1.0) and np.allclose(A_hat[3, :3], 0.0)
    eig = np.linalg.eigvalsh(normalize_adjacency(random_graph(30, 0.2, 1)))
    assert eig.max() <= 1 + 1e-9 and eig.min() >= -1 - 1e-9, "normalized adjacency has spectrum in [-1, 1]"


def test_gcn_layer_matches_message_passing_loop():
    rng = np.random.default_rng(2)
    n, d_in, d_out = 8, 5, 4
    A = random_graph(n, 0.4, 2)
    A_hat = normalize_adjacency(A)
    X = rng.normal(size=(n, d_in))
    W, b = rng.normal(size=(d_in, d_out)), rng.normal(size=d_out)
    out = gcn_layer(X, A_hat, W, b)
    assert out.shape == (n, d_out)
    ref = np.zeros((n, d_out))
    for i in range(n):
        msg = sum(A_hat[i, j] * (X[j] @ W) for j in range(n))      # aggregate messages from i's neighborhood
        ref[i] = np.maximum(msg + b, 0.0)
    assert np.allclose(out, ref, atol=1e-10)
    lin = gcn_layer(X, A_hat, W, b, activation=False)
    assert np.allclose(lin, A_hat @ X @ W + b, atol=1e-10)
    assert (lin < 0).any() and not (out < 0).any()


def test_node_embeddings_are_permutation_equivariant():
    rng = np.random.default_rng(3)
    n = 10
    A = random_graph(n, 0.3, 3)
    X = rng.normal(size=(n, 6))
    params = make_params(6, 8, 5, 3)
    H = gcn_forward(X, A, params)
    assert H.shape == (n, 5)
    perm = rng.permutation(n)
    P = np.eye(n)[perm]
    H_perm = gcn_forward(P @ X, P @ A @ P.T, params)
    assert np.allclose(H_perm, P @ H, atol=1e-10), "relabeling the nodes must relabel the embeddings"


def test_readout_is_permutation_invariant():
    rng = np.random.default_rng(4)
    n = 12
    A = random_graph(n, 0.3, 4)
    X = rng.normal(size=(n, 4))
    params = make_params(4, 8, 6, 2)
    g = graph_readout(gcn_forward(X, A, params), params)
    assert g.shape == (2,)
    for seed in range(3):
        perm = np.random.default_rng(seed).permutation(n)
        P = np.eye(n)[perm]
        g_perm = graph_readout(gcn_forward(P @ X, P @ A @ P.T, params), params)
        assert np.allclose(g_perm, g, atol=1e-10)


def test_two_layers_see_exactly_two_hops():
    rng = np.random.default_rng(5)
    n = 6
    A = path_graph(n)                                 # 0 - 1 - 2 - 3 - 4 - 5
    X = rng.normal(size=(n, 3))
    params = make_params(3, 8, 4, 2)
    H = gcn_forward(X, A, params)
    X2 = X.copy()
    X2[5] += 10.0                                     # perturb the far end
    H2 = gcn_forward(X2, A, params)
    assert np.allclose(H2[:3], H[:3], atol=1e-12), "nodes 0, 1, 2 are more than two hops from node 5"
    assert not np.allclose(H2[3:], H[3:]), "nodes within two hops must change"


def test_no_edges_reduces_to_a_per_node_mlp():
    rng = np.random.default_rng(6)
    n = 5
    X = rng.normal(size=(n, 3))
    params = make_params(3, 7, 4, 2)
    H = gcn_forward(X, np.zeros((n, n)), params)
    h1 = np.maximum(X @ params["W1"] + params["b1"], 0.0)
    ref = np.maximum(h1 @ params["W2"] + params["b2"], 0.0)
    assert np.allclose(H, ref, atol=1e-10)
    g = graph_readout(H, params)
    assert np.allclose(g, ref.mean(axis=0) @ params["W_out"] + params["b_out"], atol=1e-10)
