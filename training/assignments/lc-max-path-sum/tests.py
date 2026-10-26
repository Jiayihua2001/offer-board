import random
from collections import deque

from solution import TreeNode, max_path_sum


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


def brute_force(root):
    """Every pair of nodes (including a node with itself): sum the path through their LCA."""
    parent, nodes, stack = {root: None}, [], [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        for child in (node.left, node.right):
            if child:
                parent[child] = node
                stack.append(child)
    best = float("-inf")
    for u in nodes:
        up, total, node = {}, 0, u
        while node is not None:                  # prefix sums along the ancestors of u
            total += node.val
            up[node] = total
            node = parent[node]
        for v in nodes:
            total, node = 0, v
            while node not in up:                # climb from v until reaching an ancestor of u
                total += node.val
                node = parent[node]
            best = max(best, total + up[node])
    return best


def random_tree(rng, n, lo=-10, hi=10):
    root = TreeNode(rng.randint(lo, hi))
    slots = [(root, "left"), (root, "right")]
    for _ in range(n - 1):
        i = rng.randrange(len(slots))
        node, side = slots[i]
        slots[i] = slots[-1]
        slots.pop()
        child = TreeNode(rng.randint(lo, hi))
        setattr(node, side, child)
        slots += [(child, "left"), (child, "right")]
    return root


def test_examples():
    assert max_path_sum(build([1, 2, 3])) == 6
    assert max_path_sum(build([-10, 9, 20, None, None, 15, 7])) == 42


def test_single_node_and_all_negative():
    assert max_path_sum(TreeNode(-3)) == -3
    assert max_path_sum(build([-2, -1])) == -1
    assert max_path_sum(build([-3, -5, -4])) == -3


def test_best_path_avoids_the_root():
    assert max_path_sum(build([-100, 5, 7])) == 7
    assert max_path_sum(build([1, -2, -3, 1, 3, -2, None, -1])) == 3


def test_path_through_root_uses_both_sides():
    root = build([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])
    assert max_path_sum(root) == 48          # 7 -> 11 -> 4 -> 5 -> 8 -> 13


def test_deep_chain_is_max_subarray():
    values = [(i * 7919) % 23 - 11 for i in range(5000)]
    root = node = TreeNode(values[0])
    for i, v in enumerate(values[1:], 1):
        child = TreeNode(v)
        if i % 2:
            node.left = child
        else:
            node.right = child
        node = child
    best = cur = values[0]                   # Kadane along the chain
    for v in values[1:]:
        cur = max(v, cur + v)
        best = max(best, cur)
    assert max_path_sum(root) == best


def test_random_trees_match_brute_force():
    rng = random.Random(5)
    for _ in range(120):
        n = rng.randrange(1, 26)
        root = random_tree(rng, n)
        assert max_path_sum(root) == brute_force(root)


def test_large_tree_is_fast():
    n = 100_000
    assert max_path_sum(build([1] * n)) == 33       # two root-to-leaf arms of a 17-level complete tree
    assert max_path_sum(build([-1] * n)) == -1
