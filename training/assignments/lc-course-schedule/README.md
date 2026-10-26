# Course Schedule II

There are `num_courses` courses labelled `0` to `num_courses - 1`. `prerequisites` is a list of pairs `[a, b]` meaning you must take course `b` before course `a`.

Return an ordering of all the courses that respects every prerequisite. If several orders are valid, any one of them is accepted. If it is impossible to finish all courses (the prerequisites contain a cycle), return `[]`.

## Examples

```
find_order(2, [[1, 0]])                           -> [0, 1]
find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]])   -> [0, 1, 2, 3]  or  [0, 2, 1, 3]
find_order(2, [[1, 0], [0, 1]])                   -> []            (cycle)
find_order(1, [])                                 -> [0]
```

## Constraints

- `1 <= num_courses <= 10^5`, `0 <= len(prerequisites) <= 2 * 10^5`.
- Pairs may repeat; a pair `[a, a]` is a cycle of length one.
- Target: O(V + E) time and space. A chain of 10^5 courses is in the tests, so a recursive DFS needs an explicit stack or a raised recursion limit.

## What the interviewer listens for

- You translate the story into a directed graph and state that a valid order exists iff the graph is a DAG, before writing any code.
- Kahn's algorithm explained through its invariant: every course popped has had all of its prerequisites popped already. Cycle detection falls out for free from the length of the output.
- The DFS alternative (reverse post-order, three colours for cycle detection) and why you prefer one or the other in Python.
- Complexity O(V + E): adjacency lists rather than a matrix, and why repeated pairs do not break the in-degree count.

## Follow-up

Courses now have durations and you can take any number in parallel. What is the minimum time to finish everything? (Longest path in a DAG, computed in the same order.)
