import sys

sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    def visit(node):
        """Post-order. Returns p, q, their LCA, or None, depending on what this subtree contains."""
        if node is None or node is p or node is q:
            return node
        left = visit(node.left)
        right = visit(node.right)
        if left is not None and right is not None:
            return node                      # p and q are on different sides: this is the split point
        return left if left is not None else right

    return visit(root)
