def normalized(logp: float, n: int, alpha: float) -> float:
    return logp / max(n, 1) ** alpha


def beam_search(step_fn, start: list[int], beam_width: int, max_len: int, eos: int, alpha: float = 0.0):
    start = list(start)
    live = [(start, 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for prefix, logp in live:
            for token, lp in enumerate(step_fn(prefix)):
                candidates.append((logp + lp, prefix + [token]))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        live = []
        for logp, seq in candidates[:beam_width]:
            (finished if seq[-1] == eos else live).append((seq, logp))
        if not live:
            break
    finished.extend(live)
    seq, logp = min(finished, key=lambda f: (-normalized(f[1], len(f[0]) - len(start), alpha), f[0]))
    return seq, normalized(logp, len(seq) - len(start), alpha)
