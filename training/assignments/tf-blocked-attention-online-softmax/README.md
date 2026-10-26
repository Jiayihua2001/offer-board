# Blocked attention with online softmax (the FlashAttention idea)

Standard attention materializes the `T x T` score matrix, which at `T = 32k` is a gigabyte per head in fp32 and, worse, has to go out to GPU memory and back between the softmax and the `@ v`. FlashAttention never builds it: it streams the keys and values in blocks and keeps, per query, a running max and a running sum so that the softmax can be finished incrementally. The output is exactly the same; the memory is `O(T d)` instead of `O(T^2)`, and the whole thing lives in on-chip SRAM. Implement the algorithm in NumPy for one head.

```
blocked_attention(q, k, v, block_size) -> (out, lse)
```

- `q` is `(T_q, d)`, `k` is `(T_k, d)`, `v` is `(T_k, d_v)`, no mask. Process the keys in blocks of `block_size` rows (the last block may be shorter) and never form an array with `T_q * T_k` elements.
- For each block, compute the block of scores `s = q @ k_blk^T / sqrt(d)` of shape `(T_q, block_size)`, update the running maximum `m`, rescale the running sum `l` and the running output `acc` by `exp(m_old - m_new)`, and add the block's contribution `exp(s - m_new)` to both.
- Return `out = acc / l` of shape `(T_q, d_v)` and `lse = m + log(l)` of shape `(T_q,)`, the row-wise log-sum-exp of the scores, which is what FlashAttention stores for the backward pass instead of the weights.

## Example

```
q = k = v = I_3 (the 3x3 identity), block_size = 2
   block 1 (keys 0, 1): scores / sqrt 3, running m = [0.577, 0.577, 0], l = [1.56, 1.56, 2]
   block 2 (key 2):     m stays for rows 0, 1 and becomes 0.577 for row 2; everything is rescaled and added
out == attention(q, k, v)[0] to 1e-12, with e^(1/sqrt 3) / (e^(1/sqrt 3) + 2) = 0.4711 on the diagonal
lse == log(e^(1/sqrt 3) + 2) = [1.3300, 1.3300, 1.3300]
```

## Constraints

- Results must match full attention to `1e-8` for any `block_size`, including `1`, sizes that do not divide `T_k`, and sizes larger than `T_k`.
- Use `np.exp` for the exponentials: the tests spy on it and fail if an array with `T_q * T_k` elements is ever exponentiated. They also measure peak memory at `T = 2048` with `block_size = 64` and expect it below a quarter of one `T x T` matrix.
- Stable for scores in the thousands (the running max handles it).

## What the interviewer listens for

- Why the rescaling is exact: `exp(s - m_new) = exp(s - m_old) * exp(m_old - m_new)`, so a change of the reference max is a single multiplicative correction to the sum and the accumulated output; the result is identical to the two-pass softmax.
- The cost: the same FLOPs as standard attention (slightly more, for the rescaling), but memory traffic drops from `O(T^2)` to `O(T d)`, and since attention is memory-bound that is a 2 to 4x wall-clock win; the backward recomputes the block of weights from `q`, `k` and `lse` instead of storing them.
- The failure mode: initializing `m` to `0` instead of `-inf` (wrong when all scores are negative), or forgetting to rescale `acc` when the max changes (correct only if the max is found in the first block).
- Follow-up: adding the causal mask (skip blocks entirely above the diagonal, which halves the work), blocking over queries too so each block fits in SRAM, and how the KV cache fits into the same loop at decode time.
