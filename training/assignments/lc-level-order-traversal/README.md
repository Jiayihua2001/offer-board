# Binary Tree Level Order Traversal

Given the root of a binary tree, return its values level by level: a list of lists where the first inner list holds the root's value, the second holds the values at depth 1 from left to right, and so on. Return `[]` for an empty tree.

`TreeNode` (fields `val`, `left`, `right`) is defined in the starter; the tests build the trees for you.

## Examples

```
        3
       / \
      9  20
        /  \
       15   7

level_order(root)         -> [[3], [9, 20], [15, 7]]

level_order(TreeNode(1))  -> [[1]]
level_order(None)         -> []
```

## Constraints

- `0 <= number of nodes <= 10^5`; the tree can be a 5000-node chain.
- Values are integers and may repeat.
- Target: O(n) time and O(w) extra space, where `w` is the width of the widest level.

## What the interviewer listens for

- You name BFS right away, explain why a plain queue loses the level boundaries, and fix that with the per-level count (a sentinel or two alternating lists also work).
- You use `collections.deque`: `list.pop(0)` is O(n) per pop and turns a wide tree into O(n^2).
- You can give the DFS alternative (carry the depth, append to `out[depth]`), say when it is handy (zigzag, right-side view) and what it costs (recursion depth equal to the height).
- Complexity: every node is enqueued and dequeued once, so O(n) time; the queue never holds more than one level.

## Follow-up

Return the levels bottom-up, or in zigzag order, with a one-line change. Where does the change go?
