# A graph convolution layer and message passing

For a candidate who has trained interatomic potentials and crystal-structure models, the GNN question is less "can you do it" and more "can you explain it precisely": what the normalized adjacency does, why the layer is permutation-equivariant, why pooling makes the graph readout invariant, and what two layers can and cannot see. Implement the Kipf and Welling GCN in NumPy with dense adjacency matrices.

- `normalize_adjacency(A)`: `A` is a symmetric `(N, N)` 0/1 matrix with zero diagonal. Return `D^-1/2 (A + I) D^-1/2`, where `D` is the diagonal degree matrix of `A + I`.
- `gcn_layer(X, A_hat, W, b, activation=True)`: `relu(A_hat @ X @ W + b)` for `X (N, d_in)`, `W (d_in, d_out)`, `b (d_out,)`; skip the ReLU when `activation=False`.
- `gcn_forward(X, A, params)`: two GCN layers with ReLU after each, using `params["W1"], params["b1"], params["W2"], params["b2"]`. Returns node embeddings `H` of shape `(N, d2)`.
- `graph_readout(H, params)`: mean-pool the node embeddings over nodes, then apply a linear head `params["W_out"] (d2, d_out)` and `params["b_out"] (d_out,)`. Returns a vector of shape `(d_out,)`.

`relu` is provided in the starter.

## Example

```
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)   # path 0 - 1 - 2
normalize_adjacency(A)
  -> [[0.5,    0.408, 0.0  ],      # 1/2 on the diagonal of the end nodes (degree 2 with the self-loop)
      [0.408,  0.333, 0.408],      # 1/sqrt(2 * 3) between an end node and the middle
      [0.0,    0.408, 0.5  ]]

P = permutation matrix
gcn_forward(P @ X, P @ A @ P.T, params) == P @ gcn_forward(X, A, params)        # equivariant
graph_readout(gcn_forward(P @ X, P @ A @ P.T, params), params) == graph_readout(gcn_forward(X, A, params), params)   # invariant
```

## Constraints

- No loops over nodes or edges: the layer is matrix products and broadcasting.
- The normalization must handle isolated nodes (degree 0 before the self-loop) without division by zero.
- Dense `O(N^2 d)` per layer is fine here; be ready to say how a sparse or edge-list implementation changes it.

## What the interviewer listens for

- Why the symmetric normalization: without it, `A @ X` scales features by the degree, so high-degree nodes dominate and depth makes activations explode; `D^-1/2 (A + I) D^-1/2` has eigenvalues in `[-1, 1]`, and the self-loop keeps a node's own features in the sum.
- Equivariance as a proof, not a slogan: `P A P^T` and `P X` give `P (A X W)` because `P^T P = I`; every per-node operation commutes with `P`; summing or averaging over rows removes the `P`.
- The receptive field: after `L` layers a node has seen its `L`-hop neighborhood and nothing further (a test perturbs a far node), and over-smoothing as `L` grows.
- Follow-up: what message passing with edge features looks like (distances or bond types in an interatomic potential), why sum pooling can distinguish graphs that mean pooling cannot, and the limits of this family (the 1-WL test).
