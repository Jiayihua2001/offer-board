import math
import random

from solution import beam_search

EOS, A, B, BOS = 0, 1, 2, 3
TINY = 1e-12


def bigram_model(V, seed):
    """A deterministic toy LM: the next-token distribution depends only on the last token."""
    rng = random.Random(seed)
    table = []
    for _ in range(V):
        w = [rng.random() for _ in range(V)]
        w[EOS] = 0.05 + 0.3 * rng.random()
        z = sum(w)
        table.append([math.log(x / z) for x in w])
    return lambda prefix: table[prefix[-1]]


def garden_path(prefix):
    """A is the greedy choice but ends at once; B B B EOS is longer and better per token."""
    last = prefix[-1]
    if last == BOS:
        p = [0.02, 0.55, 0.43, TINY]
    elif last == A:
        p = [0.9, 0.05, 0.05, TINY]
    elif prefix.count(B) < 3:
        p = [0.05, 0.05, 0.9, TINY]
    else:
        p = [0.9, 0.05, 0.05, TINY]
    return [math.log(x) for x in p]


def greedy(step_fn, start, max_len, eos):
    seq, logp = list(start), 0.0
    for _ in range(max_len):
        lp = step_fn(seq)
        t = max(range(len(lp)), key=lambda i: lp[i])
        seq.append(t)
        logp += lp[t]
        if t == eos:
            break
    return seq, logp


def path_logp(step_fn, start, tokens):
    total = 0.0
    for i in range(len(start), len(tokens)):
        total += step_fn(tokens[:i])[tokens[i]]
    return total


def test_beam_width_one_is_greedy():
    for seed in range(6):
        step_fn = bigram_model(6, seed)
        for start in ([1], [5], [2, 3]):
            tokens, score = beam_search(step_fn, start, 1, 10, EOS)
            g_tokens, g_logp = greedy(step_fn, start, 10, EOS)
            assert tokens == g_tokens, (seed, start, tokens, g_tokens)
            assert math.isclose(score, g_logp, abs_tol=1e-9)


def test_worked_example():
    tokens, score = beam_search(garden_path, [BOS], 1, 8, EOS)
    assert tokens == [BOS, A, EOS] and math.isclose(score, math.log(0.55 * 0.9), abs_tol=1e-9)
    tokens, score = beam_search(garden_path, [BOS], 2, 8, EOS, alpha=0.0)
    assert tokens == [BOS, A, EOS] and math.isclose(score, math.log(0.55 * 0.9), abs_tol=1e-9)
    tokens, score = beam_search(garden_path, [BOS], 2, 8, EOS, alpha=1.0)
    assert tokens == [BOS, B, B, B, EOS], tokens
    assert math.isclose(score, math.log(0.43 * 0.9 ** 3) / 4, abs_tol=1e-9)


def test_length_penalty_changes_the_winner_monotonically():
    lengths = []
    for alpha in (0.0, 0.5, 1.0, 2.0):
        tokens, _ = beam_search(garden_path, [BOS], 3, 8, EOS, alpha=alpha)
        lengths.append(len(tokens))
    assert lengths == sorted(lengths), lengths
    assert lengths[0] < lengths[-1]


def test_finished_hypotheses_are_not_extended_and_eos_ends_the_sequence():
    seen = []

    def spy(prefix):
        seen.append(list(prefix))
        return garden_path(prefix)

    tokens, _ = beam_search(spy, [BOS], 3, 8, EOS)
    assert tokens[-1] == EOS and EOS not in tokens[:-1]
    assert all(p[-1] != EOS for p in seen), "step_fn was called on a finished hypothesis"
    assert all(p[0] == BOS for p in seen)


def test_max_len_bounds_the_output():
    def never_ends(prefix):
        return [-40.0, math.log(0.6), math.log(0.4), -40.0]

    tokens, score = beam_search(never_ends, [BOS], 2, 5, EOS)
    assert len(tokens) == 1 + 5 and EOS not in tokens
    assert tokens == [BOS, A, A, A, A, A]
    assert math.isclose(score, 5 * math.log(0.6), abs_tol=1e-9)
    # a longer budget allows a longer output; the per-token score is the same with alpha = 1
    tokens7, score7 = beam_search(never_ends, [BOS], 2, 7, EOS, alpha=1.0)
    assert len(tokens7) == 8 and math.isclose(score7, math.log(0.6), abs_tol=1e-9)


def test_score_matches_the_path_under_the_model():
    for seed in range(4):
        step_fn = bigram_model(7, seed)
        for width, alpha in ((2, 0.0), (3, 0.7), (5, 1.0)):
            tokens, score = beam_search(step_fn, [4], width, 9, EOS, alpha=alpha)
            n = len(tokens) - 1
            assert tokens[0] == 4 and 1 <= n <= 9
            assert math.isclose(score, path_logp(step_fn, [4], tokens) / n ** alpha, abs_tol=1e-9), (seed, width, alpha)


def test_wider_beams_never_score_worse_here_and_sometimes_beat_greedy():
    improved = 0
    for seed in range(8):
        step_fn = bigram_model(6, seed)
        scores = [beam_search(step_fn, [3], w, 10, EOS)[1] for w in (1, 2, 4, 8)]
        for narrow, wide in zip(scores, scores[1:]):
            assert wide >= narrow - 1e-9, (seed, scores)
        improved += scores[-1] > scores[0] + 1e-9
    assert improved >= 1, "a beam of 8 should beat greedy on at least one of these models"
