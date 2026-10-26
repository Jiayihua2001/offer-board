# LRU Cache

Design a fixed-capacity cache that evicts the least recently used entry when it is full. Implement a class `LRUCache(capacity)` with two methods, both **O(1)** on average:

- `get(key) -> int`: return the value stored for `key`, or `-1` if it is not in the cache. A successful `get` makes that key the most recently used.
- `put(key, value) -> None`: insert the pair, or update the value if `key` is already present. Either way the key becomes the most recently used. If inserting a new key would exceed `capacity`, evict the least recently used key first.

"Used" means touched by a successful `get` or by any `put`. A `get` of a missing key changes nothing. Keys and values are integers and `capacity >= 1`.

Write it with a hash map plus your own doubly linked list; interviewers at the companies you are targeting usually say no to `OrderedDict` and `functools.lru_cache`, though the tests accept any correct implementation.

## Examples

```
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
c.get(1)      -> 1        (1 is now the most recent, 2 the least)
c.put(3, 3)               (evicts 2)
c.get(2)      -> -1
c.put(4, 4)               (evicts 1)
c.get(1)      -> -1
c.get(3)      -> 3
c.get(4)      -> 4
```

```
c = LRUCache(2)
c.put(1, 1); c.put(2, 2); c.put(1, 10)   (1 updated and refreshed)
c.put(3, 3)                              (evicts 2, not 1)
c.get(1)      -> 10
c.get(2)      -> -1
```

## Constraints

- `1 <= capacity <= 2 * 10^5`, up to `5 * 10^5` operations.
- Target: O(1) per operation, O(capacity) space. Any O(capacity) scan per operation (a list `remove`, `index` or `pop(0)`) will time out on the large test.

## What the interviewer listens for

- You decompose the requirement into the three primitive operations and explain why a dict alone or a list alone cannot deliver all of them in O(1).
- Why the list must be doubly linked: you reach a node through the dict and must unlink it without walking from the head.
- Sentinels: a dummy head and tail mean you never test for an empty list or a `None` neighbour, which is where most bugs live under time pressure.
- You walk through both refresh paths (a `get` and a `put` of an existing key) and the eviction path on a two-element example before declaring victory.

## Follow-up to think about

How does LFU differ, and why is O(1) LFU much harder? What changes if the cache must be safe across threads, or entries have TTLs?
