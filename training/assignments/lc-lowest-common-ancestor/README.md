# Lowest Common Ancestor of a Binary Tree

Given the root of a binary tree and two nodes `p` and `q` in it, return their lowest common ancestor: the deepest node that has both `p` and `q` as descendants, where a node counts as a descendant of itself.

`p` and `q` are `TreeNode` objects (not values), they are distinct, and both are guaranteed to be in the tree. Values are unique. Return the `TreeNode`, not its value. The tree is not a search tree and nodes have no parent pointers.

## Examples

```
            3
          /   \
         5     1
        / \   / \
       6   2 0   8
          / \
         7   4

lowest_common_ancestor(root, node(5), node(1))  -> node(3)
lowest_common_ancestor(root, node(5), node(4))  -> node(5)   (5 is an ancestor of 4)
lowest_common_ancestor(root, node(7), node(4))  -> node(2)
```

## Constraints

- `2 <= number of nodes <= 10^5`; the tree can be a 5000-node chain, so a recursive solution should raise the recursion limit (or use an explicit stack).
- Target: O(n) time, O(h) extra space where `h` is the height.

## What the interviewer listens for

- The brute force: record the root-to-`p` and root-to-`q` paths (O(n) each) and take the last node they share. It is already O(n); the recursive version is about doing it in one pass with O(h) space and no extra lists.
- The post-order invariant: each call returns `p`, `q`, their LCA, or `None`, and you can say why "both sides non-None means this node is the answer" is correct.
- Why returning early when `root is p or root is q` is valid without looking below it: the other node is either underneath (then this node is the answer) or elsewhere (then the parent will see both sides).
- What changes if the tree is a BST, if nodes have parent pointers, or if `p` or `q` might be missing from the tree.

## Follow-up

With parent pointers, how do you find the LCA in O(h) time and O(1) extra space?
