# Binary Tree Maximum Path Sum

A path in a binary tree is a sequence of nodes in which consecutive nodes are joined by an edge and no node appears twice. It does not need to pass through the root and it may consist of a single node. Its sum is the sum of the values on it.

Given the root of a non-empty binary tree, return the largest sum of any path. Values may be negative.

## Examples

```
    1
   / \
  2   3           max_path_sum(root) -> 6     (2 -> 1 -> 3)

   -10
   /  \
  9   20
     /  \
    15   7        max_path_sum(root) -> 42    (15 -> 20 -> 7)

max_path_sum(TreeNode(-3)) -> -3             (a path has at least one node)
```

## Constraints

- `1 <= number of nodes <= 10^5`, values in `[-1000, 1000]`.
- The tree can be a 5000-node chain, so use an explicit stack or raise the recursion limit.
- Target: O(n) time, O(h) extra space.

## What the interviewer listens for

- You separate two quantities: the best path that **turns** at a node (may use both children) and the best path that **continues upward** from it (may use only one child). Mixing them up is the classic bug.
- Negative subtrees are dropped with `max(0, gain)`, but a single negative node is still a valid answer, so the running best starts at minus infinity, not at 0.
- Brute force first: for every pair of nodes compute the path sum through their LCA, O(n^2) or worse; then why one post-order pass is enough.
- Complexity O(n) time; the stack depth equals the height, and what you do about that in Python.

## Follow-up

How would you return the path itself, not just its sum?
