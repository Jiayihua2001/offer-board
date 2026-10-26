from collections import Counter


def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    return dict(Counter(zip(ids, ids[1:])))


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    a, b = pair
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text: str, num_merges: int) -> list[tuple[int, int]]:
    ids = list(text.encode("utf-8"))
    merges = []
    for step in range(num_merges):
        counts = pair_counts(ids)
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))  # most frequent, then lexicographically smallest
        merges.append(best)
        ids = merge_pair(ids, best, 256 + step)
    return merges


def build_vocab(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        vocab[256 + i] = vocab[a] + vocab[b]
    return vocab


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    ids = list(text.encode("utf-8"))
    for i, pair in enumerate(merges):
        if len(ids) < 2:
            break
        ids = merge_pair(ids, pair, 256 + i)
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
