"""Byte-level byte-pair encoding: trainer, encoder, decoder. Standard library only."""


def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Counts of every adjacent pair (ids[i], ids[i + 1]). Overlapping occurrences all count."""
    raise NotImplementedError


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` by `new_id`, scanning left to right without overlap."""
    raise NotImplementedError


def train_bpe(text: str, num_merges: int) -> list[tuple[int, int]]:
    """Learn up to `num_merges` merges over the UTF-8 bytes of `text`.

    Each step merges the most frequent pair (ties: the lexicographically smallest pair) into the
    id 256 + step. Stop early when no pair is left. Returns the merges in order.
    """
    raise NotImplementedError


def build_vocab(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """id -> bytes for the 256 byte values plus one entry per merge."""
    raise NotImplementedError


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Token ids for `text`: its UTF-8 bytes with the merges applied in training order."""
    raise NotImplementedError


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Concatenate the bytes of each id and decode as UTF-8 (errors="replace")."""
    raise NotImplementedError
