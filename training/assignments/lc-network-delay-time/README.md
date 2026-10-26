# Network Delay Time

You are given a network of `n` nodes labelled `1` to `n` and a list `times` of directed edges `[u, v, w]`: a signal sent from `u` reaches `v` after `w` time units. At time 0 a signal is sent from node `k` and spreads along every edge.

Return the time at which all `n` nodes have received the signal, or `-1` if some node never does.

## Examples

```
network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], n=4, k=2)  -> 2
network_delay_time([[1, 2, 1]], n=2, k=1)                        -> 1
network_delay_time([[1, 2, 1]], n=2, k=2)                        -> -1     (node 1 is unreachable)
network_delay_time([], n=1, k=1)                                 -> 0
```

## Constraints

- `1 <= k <= n <= 2 * 10^4`, `0 <= len(times) <= 10^5`, so the dense O(n^2) version is too slow.
- `0 <= w <= 100`; parallel edges with different weights can exist; cycles can exist.
- Target: O(E log E) time with a binary heap, O(V + E) space.

## What the interviewer listens for

- You name the problem: single-source shortest paths with non-negative weights, so Dijkstra; and the answer is the maximum over the shortest distances, which is a different thing from "the longest chain of edges".
- Lazy deletion: `heapq` has no decrease-key, so you push duplicates and skip stale entries when they are popped. You can say why that is still correct and what it costs (O(E log E)).
- The unreachable case comes from counting settled nodes or from scanning the distance array for infinity afterwards; either is fine, but you handle it.
- When Bellman-Ford or plain BFS would be the right tool instead (negative weights; unit weights), and the O(V^2) array version for dense graphs.

## Follow-up

Each node `v` now adds a processing delay `p[v]` before forwarding the signal. Where does that go in your algorithm?
