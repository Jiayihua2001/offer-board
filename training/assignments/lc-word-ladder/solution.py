from collections import defaultdict, deque


def ladder_length(begin_word: str, end_word: str, word_list: list[str]) -> int:
    words = set(word_list)
    if end_word not in words:
        return 0
    L = len(begin_word)
    buckets = defaultdict(list)                  # "h*t" -> every word matching the pattern
    for w in words:
        for i in range(L):
            buckets[w[:i] + "*" + w[i + 1:]].append(w)

    seen = {begin_word}
    queue = deque([(begin_word, 1)])
    while queue:
        word, steps = queue.popleft()
        if word == end_word:
            return steps
        for i in range(L):
            pattern = word[:i] + "*" + word[i + 1:]
            for nxt in buckets[pattern]:
                if nxt not in seen:
                    seen.add(nxt)                # mark on enqueue, not on dequeue
                    queue.append((nxt, steps + 1))
            buckets[pattern] = []                # every member is now seen; never scan this bucket again
    return 0
