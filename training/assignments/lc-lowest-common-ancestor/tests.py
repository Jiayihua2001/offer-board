import random
from collections import deque

from solution import TreeNode, lowest_common_ancestor


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


def find(root, val):
    """Return the node holding val (values are unique)."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.val == val:
            return node
        stack.extend(child for child in (node.left, node.right) if child)
    raise AssertionError(f"no node with value {val}")


def brute_lca(root, p, q):
    """Independent answer: parent pointers, then walk q up until it hits an ancestor of p."""
    parent = {root: None}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in (node.left, node.right):
            if child:
                parent[child] = node
                stack.append(child)
    ancestors = set()
    node = p
    while node is not None:
        ancestors.add(node)
        node = parent[node]
    node = q
    while node not in ancestors:
        node = parent[node]
    return node


def random_tree(rng, n):
    """A random-shaped tree with values 0..n-1."""
    root = TreeNode(0)
    slots = [(root, "left"), (root, "right")]
    for val in range(1, n):
        i = rng.randrange(len(slots))
        parent, side = slots[i]
        slots[i] = slots[-1]
        slots.pop()
        child = TreeNode(val)
        setattr(parent, side, child)
        slots += [(child, "left"), (child, "right")]
    return root


def check(root, a, b, expected):
    got = lowest_common_ancestor(root, find(root, a), find(root, b))
    assert isinstance(got, TreeNode), f"expected a TreeNode, got {got!r}"
    assert got.val == expected, f"LCA({a}, {b}) should be {expected}, got {got.val}"


EXAMPLE = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]


def test_example():
    root = build(EXAMPLE)
    check(root, 5, 1, 3)
    check(root, 7, 4, 2)
    check(root, 6, 8, 3)


def test_one_node_is_an_ancestor_of_the_other():
    root = build(EXAMPLE)
    check(root, 5, 4, 5)
    check(root, 3, 8, 3)        # the root itself
    check(root, 2, 7, 2)


def test_order_of_p_and_q_does_not_matter():
    root = build(EXAMPLE)
    check(root, 4, 5, 5)
    check(root, 0, 7, 3)
    check(root, 1, 5, 3)


def test_two_node_trees():
    check(build([1, 2]), 1, 2, 1)
    check(build([1, None, 2]), 2, 1, 1)


def test_deep_chain():
    root = node = TreeNode(0)
    for i in range(1, 5000):
        child = TreeNode(i)
        if i % 3 == 0:
            node.right = child
        else:
            node.left = child
        node = child
    check(root, 4999, 4000, 4000)
    check(root, 2500, 4999, 2500)


def test_random_trees_match_brute_force():
    rng = random.Random(11)
    for _ in range(150):
        n = rng.randrange(2, 60)
        root = random_tree(rng, n)
        a, b = rng.sample(range(n), 2)
        p, q = find(root, a), find(root, b)
        got = lowest_common_ancestor(root, p, q)
        want = brute_lca(root, p, q)
        assert got is want, f"LCA({a}, {b}) should be {want.val}, got {getattr(got, 'val', got)}"


def test_large_tree_is_fast():
    n = 100_000
    root = build(list(range(n)))         # complete binary tree
    p, q = find(root, n - 1), find(root, 12_345)
    want = brute_lca(root, p, q)
    got = lowest_common_ancestor(root, p, q)
    assert got is want, f"expected {want.val}, got {getattr(got, 'val', got)}"
