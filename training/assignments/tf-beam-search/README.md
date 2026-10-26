# Beam search with length normalization

Greedy decoding commits to the best next token and can walk into a dead end: a likely first word followed by an unlikely continuation. Beam search keeps the `k` best partial sequences instead and only commits at the end. It is the standard decoder for translation and summarization, and the thing you reach for when a sampling-based decoder is too noisy. Write it against an abstract model so the search logic is all that matters.

```
beam_search(step_fn, start, beam_width, max_len, eos, alpha=0.0) -> (tokens, score)
```

- `step_fn(prefix: list[int]) -> list[float]`: log-probabilities of the next token given the prefix (a deterministic toy model in the tests).
- `start`: the initial tokens (for instance `[BOS]`). The returned `tokens` begin with `start`.
- The algorithm, exactly: keep a list of live hypotheses, initially `[(start, 0.0)]`. Up to `max_len` rounds: extend every live hypothesis with every possible next token, rank all candidates by their total log-probability, keep the best `beam_width`. A kept candidate that ends in `eos` is finished and never extended again; the rest stay live. Stop when nothing is live. After the last round, anything still live counts as finished.
- Return the finished hypothesis with the highest length-normalized score `logp / n ** alpha`, where `n` is the number of generated tokens (`len(tokens) - len(start)`), and that score. `alpha = 0` is the raw log-probability; `alpha = 1` is the per-token average.

## Example

```
EOS = 0, A = 1, B = 2, BOS = 3, with a model where
   after BOS:  p(EOS) = 0.02, p(A) = 0.55, p(B) = 0.43
   after A:    p(EOS) = 0.9
   after B:    p(B) = 0.9 until three B's have been produced, then p(EOS) = 0.9
beam_search(step_fn, [BOS], 1, 8, EOS)            -> ([3, 1, 0], log(0.55 * 0.9) = -0.703)   (greedy)
beam_search(step_fn, [BOS], 2, 8, EOS, alpha=0)   -> ([3, 1, 0], -0.703)
beam_search(step_fn, [BOS], 2, 8, EOS, alpha=1)   -> ([3, 2, 2, 2, 0], log(0.43 * 0.9^3) / 4 = -0.290)
```

## Constraints

- Pure Python, no NumPy: a few hundred `step_fn` calls per search at most.
- Never call `step_fn` on a prefix that already ends in `eos`.
- With `beam_width = 1` the result must equal greedy decoding, token for token.

## What the interviewer listens for

- Why beam search is not exact: it is a pruned breadth-first search, so a wider beam usually but not always finds a better sequence, and the model's probability is not the same thing as the quality you want anyway.
- The cost: `beam_width * max_len` model calls, which with a KV cache means keeping `beam_width` caches and re-indexing them when beams are reordered; batching the beams into one forward pass is the standard implementation.
- The failure mode: without length normalization every extra token multiplies by a probability below one, so the search prefers short outputs; `alpha` (or the GNMT `((5 + n) / 6) ** alpha` penalty) corrects that, and beams that are too wide produce bland, repetitive text, which is why chat models sample instead.
- Follow-up: how to add an n-gram repetition block or a coverage term, and what diverse beam search changes.
