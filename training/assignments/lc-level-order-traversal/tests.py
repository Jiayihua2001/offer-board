import random
from collections import deque

from solution import TreeNode, level_order


def build(values):
    """Build a tree from a LeetCode-style level-order list; None marks a missing child."""
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def reference(root):
    """Independent answer: pre-order DFS on an explicit stack, bucketing values by depth."""
    out = []
    stack = [(root, 0)] if root else []
    while stack:
        node, depth = stack.pop()
        if depth == len(out):
            out.append([])
        out[depth].append(node.val)
        if node.right:
            stack.append((node.right, depth + 1))
        if node.left:
            stack.append((node.left, depth + 1))
    return out


def test_example():
    root = build([3, 9, 20, None, None, 15, 7])
    assert level_order(root) == [[3], [9, 20], [15, 7]]


def test_empty_and_single_node():
    assert level_order(None) == []
    assert level_order(TreeNode(1)) == [[1]]


def test_ragged_levels_keep_left_to_right_order():
    #        1
    #      /   \
    #     2     3
    #      \   /
    #       4 5
    #      /   \
    #     6     7
    root = build([1, 2, 3, None, 4, 5, None, 6, None, None, 7])
    assert level_order(root) == [[1], [2, 3], [4, 5], [6, 7]]


def test_duplicate_and_negative_values():
    root = build([5, 5, -5, 5, None, None, -5])
    assert level_order(root) == [[5], [5, -5], [5, -5]]


def test_deep_chain():
    root = node = TreeNode(0)
    for i in range(1, 5000):
        child = TreeNode(i)
        if i % 2:
            node.left = child
        else:
            node.right = child
        node = child
    assert level_order(root) == [[i] for i in range(5000)]


def test_random_trees_match_reference():
    rng = random.Random(2027)
    for _ in range(200):
        n = rng.randrange(0, 40)
        values = [rng.randrange(-5, 6) if rng.random() < 0.8 else None for _ in range(n)]
        root = build(values)
        assert level_order(root) == reference(root), f"wrong levels for {values}"


def test_wide_tree_is_fast():
    n = 100_000
    root = build(list(range(n)))        # complete binary tree, last level 34,465 wide
    out = level_order(root)
    assert len(out) == 17, f"expected 17 levels, got {len(out)}"
    assert [len(level) for level in out[:3]] == [1, 2, 4]
    assert sum(len(level) for level in out) == n
    assert out[-1][0] == 2 ** 16 - 1 and out[-1][-1] == n - 1
