"""Lowest Common Ancestor. Keep the names; the tests import lowest_common_ancestor and TreeNode."""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """Return the deepest node that has both p and q in its subtree (a node is its own descendant)."""
    raise NotImplementedError
