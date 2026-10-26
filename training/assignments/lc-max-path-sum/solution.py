import sys

sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum(root: TreeNode) -> int:
    best = float("-inf")

    def gain(node):
        """Best sum of a path that starts at node and goes down one side (or stops here)."""
        nonlocal best
        if node is None:
            return 0
        left = max(0, gain(node.left))       # a negative branch is better left out
        right = max(0, gain(node.right))
        best = max(best, node.val + left + right)    # the path that turns at this node
        return node.val + max(left, right)   # only one branch may continue upward

    gain(root)
    return best
