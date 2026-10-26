"""Binary Tree Level Order Traversal. Keep the names; the tests import level_order and TreeNode."""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root: TreeNode | None) -> list[list[int]]:
    """Return the node values level by level, each level from left to right; [] for an empty tree."""
    raise NotImplementedError
