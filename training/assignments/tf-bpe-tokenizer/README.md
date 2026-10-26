# Byte-pair encoding from scratch

Everything a language model sees starts here. A BPE tokenizer turns text into integer ids by starting from raw bytes and repeatedly merging the most frequent adjacent pair into a new symbol. GPT-2, Llama and most modern models use exactly this idea (plus a regex pre-split). You will write the trainer, the encoder and the decoder: byte level, with no pre-tokenization.

Symbols are integers: the 256 byte values first, then one new id per merge (`256 + i` for merge number `i`).

- `pair_counts(ids) -> dict[(int, int), int]`: counts of every adjacent pair `(ids[i], ids[i+1])`. `[97, 97, 97, 98]` gives `{(97, 97): 2, (97, 98): 1}`; overlapping occurrences all count.
- `merge_pair(ids, pair, new_id) -> list[int]`: replace every occurrence of `pair` by `new_id`, scanning left to right without overlap. `[97, 97, 97, 98]` with pair `(97, 97)` becomes `[256, 97, 98]`.
- `train_bpe(text, num_merges) -> list[(int, int)]`: start from `list(text.encode("utf-8"))`, then `num_merges` times: count pairs, pick the most frequent pair (ties go to the lexicographically smallest pair), record it, merge it everywhere. Stop early when no pair is left. Return the merges in order.
- `build_vocab(merges) -> dict[int, bytes]`: id to bytes for the 256 byte values plus one entry per merge.
- `encode(text, merges) -> list[int]`: apply the merges, in training order, to the bytes of `text`.
- `decode(ids, vocab) -> str`: concatenate the bytes and decode as UTF-8 with `errors="replace"`, since a token boundary can split a multi-byte character.

## Example

```
merges = train_bpe("aaabdaaabac", 3)    -> [(97, 97), (97, 98), (256, 257)]
    aaabdaaabac -> ZabdZabac (Z = 256) -> ZYdZYac (Y = 257) -> XdXac (X = 258)
encode("aaabdaaabac", merges)           -> [258, 100, 258, 97, 99]
build_vocab(merges)[258]                -> b"aaab"
decode([258, 100], build_vocab(merges)) -> "aaabd"
```

In the second step `(256, 97)` and `(97, 98)` both occur twice; the lexicographically smaller `(97, 98)` wins.

## Constraints

- Byte level: any string, in any script or emoji, must round-trip exactly, even if it never appeared in training.
- Deterministic: the same text and `num_merges` always give the same merges.
- No pre-tokenization in this version, so merges may cross spaces. Training on a few kilobytes with 40 merges must take well under a second.

## What the interviewer listens for

- Why bytes and not characters: a 256-symbol base means there is never an unknown token, at the price of up to 4 tokens for one rare character.
- The cost: training as written is `O(num_merges * len(text))` and encoding is `O(len(merges) * len(text))`; the production version ranks the merges and repeatedly merges the lowest-ranked pair present, with a priority queue or a linked list, so encoding is near linear.
- The failure mode: merging across whitespace and punctuation produces tokens like `dog.`, which is why GPT-2 pre-splits with a regex so merges never cross word boundaries, and why digits get chunked oddly.
- Follow-up: how vocabulary size trades sequence length against embedding-table size and rare-token under-training, and what changes in SentencePiece's unigram model.
