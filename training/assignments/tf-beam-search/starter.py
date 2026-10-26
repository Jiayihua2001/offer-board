"""Beam search over an abstract model. Pure Python."""


def beam_search(step_fn, start: list[int], beam_width: int, max_len: int, eos: int, alpha: float = 0.0):
    """Return (tokens, score) for the best hypothesis.

    step_fn(prefix) -> list of next-token log-probabilities. `tokens` starts with `start`.
    Each round expands every live hypothesis with every token, keeps the `beam_width` best by total log-prob,
    and retires the ones ending in `eos`. Stops when nothing is live or after `max_len` rounds (leftovers count
    as finished). The winner maximizes logp / n ** alpha, with n the number of generated tokens; return that score.
    """
    raise NotImplementedError
