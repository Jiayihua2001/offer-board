# Sampling strategies and a KV cache

Inference is a different program from training. Decoding happens one token at a time, each new token needs the keys and values of every previous token, and recomputing them from scratch at every step makes generation quadratic in the output length. The KV cache fixes that, and it is the data structure every serving system is built around. On top of it sit the sampling knobs: temperature, top-k and nucleus (top-p) sampling.

The starter provides the complete NumPy GPT from the previous assignment (`init_params`, `gpt_forward`, `block`, and the helpers `causal_mask`, `attention`, `split_heads`, `merge_heads`, `layer_norm`, `gelu`, `ffn`). Write three functions:

- `sample_next(logits, rng, temperature=1.0, top_k=None, top_p=None) -> int`: one token id from a `(vocab,)` vector of logits. `temperature == 0` means greedy (argmax). Otherwise divide the logits by the temperature, then if `top_k` is set keep only the `k` largest logits, then if `top_p` is set keep the smallest set of most-probable tokens whose cumulative probability is at least `p`; renormalize and draw with `rng` (a `np.random.Generator`).
- `forward_with_cache(tokens, params, cache=None) -> (logits, cache)`: `tokens` is a 1-D array of only the new tokens; `cache` is whatever structure you choose (`None` to start). Returns logits `(len(tokens), vocab)` for the new positions and the updated cache. The result must equal the corresponding rows of `gpt_forward` on the whole sequence, whether the tokens arrive one at a time or in chunks (prefill).
- `generate(tokens, params, n_new, rng=None, use_cache=True, temperature=0.0, top_k=None, top_p=None) -> list[int]`: the prompt followed by `n_new` sampled tokens. With `use_cache=False`, call `gpt_forward` on the whole sequence at each step; with the cache, prefill the prompt once and then feed one token per step. The two paths must produce identical tokens.

## Example

```
logits = log([0.5, 0.3, 0.15, 0.05])
sample_next(logits, rng, temperature=0)            -> 0 always
sample_next(logits, rng, top_k=2)                  -> 0 or 1, with probabilities 0.625 and 0.375
sample_next(logits, rng, top_p=0.75)               -> 0 or 1 (0.5 + 0.3 = 0.8 >= 0.75; 0.5 alone is not enough)
sample_next(logits, rng, top_p=0.9)                -> 0, 1 or 2
generate([3, 1, 4, 1, 5], params, 8, use_cache=True) == generate([3, 1, 4, 1, 5], params, 8, use_cache=False)
```

## Constraints

- The cached and uncached logits must agree to `1e-8`, including a prefill in chunks of several tokens.
- The cache must grow with the sequence: positions of new tokens continue from where the cache ends, so `wpe` is indexed with the absolute position.
- `sample_next` returns a Python `int`, and `rng` may be `None` when `temperature == 0` (greedy decoding never draws).

## What the interviewer listens for

- Why the cache works at all: with a causal mask the key and value of a past token never change when later tokens arrive, so `K` and `V` are append-only; the queries of past tokens are never needed again.
- The cost: the cache holds `2 * n_layers * T * d_model` values per sequence (`2 * L * T * H_kv * d_head` with grouped-query attention), which is what limits batch size on a serving GPU; decoding is then memory-bound, which is why speculative decoding and paged caches exist.
- The failure mode: forgetting the position offset for the new token (it gets position 0 again) or applying top-p before the temperature, which quietly changes the nucleus.
- Follow-up: how top-p differs from top-k when the distribution is flat versus peaked, and what min-p and repetition penalties do to the logits.
