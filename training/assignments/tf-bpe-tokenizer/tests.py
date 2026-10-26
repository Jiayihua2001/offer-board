from solution import build_vocab, decode, encode, merge_pair, pair_counts, train_bpe

CORPUS = (
    "the quick brown fox jumps over the lazy dog. the dog sleeps while the fox runs. "
    "low lower lowest; new newer newest; wide wider widest. "
    "materials science studies how the structure of a material controls its properties. "
) * 6


def test_pair_counts_and_merge_pair():
    ids = [97, 97, 97, 98]
    counts = pair_counts(ids)
    assert counts[(97, 97)] == 2 and counts[(97, 98)] == 1, counts
    assert merge_pair(ids, (97, 97), 256) == [256, 97, 98]  # left to right, no overlap
    assert merge_pair([1, 2, 1, 2, 1], (1, 2), 9) == [9, 9, 1]
    assert merge_pair([1, 1, 1, 1, 1], (1, 1), 9) == [9, 9, 1]
    assert merge_pair([5], (5, 5), 9) == [5]
    assert merge_pair([], (5, 5), 9) == []


def test_classic_example_merges_and_ids():
    merges = train_bpe("aaabdaaabac", 3)
    assert merges == [(97, 97), (97, 98), (256, 257)], merges
    assert encode("aaabdaaabac", merges) == [258, 100, 258, 97, 99]
    vocab = build_vocab(merges)
    assert len(vocab) == 259
    assert vocab[97] == b"a" and vocab[256] == b"aa" and vocab[257] == b"ab" and vocab[258] == b"aaab"
    assert decode([258, 100], vocab) == "aaabd"


def test_ties_go_to_the_lexicographically_smallest_pair():
    assert train_bpe("xyxyabab", 2) == [(97, 98), (120, 121)]
    # "the the the cat": (t,h), (h,e) and (e,' ') all occur 3 times; (e,' ') = (101, 32) is the smallest
    merges = train_bpe("the the the cat", 3)
    assert merges == [(101, 32), (104, 256), (116, 257)], merges
    assert encode("the ", merges) == [258]


def test_training_stops_when_nothing_is_left():
    assert train_bpe("ab", 10) == [(97, 98)]
    assert train_bpe("a", 3) == []
    assert train_bpe("", 3) == []


def test_merges_apply_in_training_order():
    merges = [(97, 98), (256, 99)]  # ab -> 256, then (256, c) -> 257
    assert encode("abc", merges) == [257]
    assert encode("abcabc", merges) == [257, 257]
    assert encode("bc", merges) == [98, 99]  # (98, 99) was never a merge
    assert encode("", merges) == []


def test_round_trip_including_unseen_characters():
    merges = train_bpe(CORPUS, 40)
    assert len(merges) == 40
    vocab = build_vocab(merges)
    for s in ["the cat sat on the mat", "héllo wörld 🙂", "", "completely new words zzz", "日本語のテキスト"]:
        ids = encode(s, merges)
        assert all(0 <= i < 256 + 40 for i in ids)
        assert decode(ids, vocab) == s, s
    # bytes above 127 never appeared in training, so an emoji stays four single-byte tokens
    assert encode("🙂", merges) == list("🙂".encode("utf-8"))


def test_compresses_the_training_text():
    merges = train_bpe(CORPUS, 40)
    n_bytes = len(CORPUS.encode("utf-8"))
    n_tokens = len(encode(CORPUS, merges))
    assert n_tokens < 0.6 * n_bytes, f"{n_tokens} tokens for {n_bytes} bytes"
    vocab = build_vocab(merges)
    for i, (a, b) in enumerate(merges):
        assert vocab[256 + i] == vocab[a] + vocab[b]
