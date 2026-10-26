import random
from collections import OrderedDict

from solution import LRUCache


class ReferenceLRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


def test_example():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)
    assert c.get(2) == -1, "2 was least recently used and must have been evicted"
    c.put(4, 4)
    assert c.get(1) == -1, "1 was least recently used after get(1), put(3), so it goes next"
    assert c.get(3) == 3
    assert c.get(4) == 4


def test_capacity_one():
    c = LRUCache(1)
    assert c.get(1) == -1
    c.put(1, 1)
    assert c.get(1) == 1
    c.put(2, 2)
    assert c.get(1) == -1
    assert c.get(2) == 2
    c.put(2, 22)
    assert c.get(2) == 22


def test_get_refreshes_recency():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)
    assert c.get(2) == -1, "get(1) made 2 the least recently used"
    assert c.get(1) == 1
    assert c.get(3) == 3


def test_put_existing_key_updates_value_and_refreshes():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.put(1, 10)
    c.put(3, 3)
    assert c.get(1) == 10, "updated value must be returned"
    assert c.get(2) == -1, "put(1, 10) refreshed 1, so 2 was evicted"
    assert c.get(3) == 3


def test_missing_get_does_not_change_recency():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(5) == -1
    c.put(3, 3)
    assert c.get(1) == -1, "1 was still the least recently used"
    assert c.get(2) == 2


def test_exact_eviction_order_over_a_longer_sequence():
    c = LRUCache(3)
    c.put(1, 1)
    c.put(2, 2)
    c.put(3, 3)          # recency: 1 < 2 < 3
    assert c.get(1) == 1  # 2 < 3 < 1
    c.put(4, 4)           # evicts 2 -> 3 < 1 < 4
    assert c.get(2) == -1
    assert c.get(3) == 3  # 1 < 4 < 3
    c.put(5, 5)           # evicts 1 -> 4 < 3 < 5
    assert c.get(1) == -1
    c.put(4, 44)          # 3 < 5 < 4
    c.put(6, 6)           # evicts 3 -> 5 < 4 < 6
    assert c.get(3) == -1
    assert c.get(5) == 5
    assert c.get(4) == 44
    assert c.get(6) == 6


def test_random_against_ordered_dict_reference():
    rng = random.Random(103)
    for capacity in (1, 2, 5, 50):
        c, ref = LRUCache(capacity), ReferenceLRU(capacity)
        for step in range(5000):
            key = rng.randrange(0, capacity + 10)
            if rng.random() < 0.5:
                assert c.get(key) == ref.get(key), f"capacity {capacity}, step {step}, get({key})"
            else:
                value = rng.randrange(0, 1000)
                c.put(key, value)
                ref.put(key, value)


def test_large_capacity_and_many_operations_are_fast():
    rng = random.Random(107)
    capacity = 200_000
    c, ref = LRUCache(capacity), ReferenceLRU(capacity)
    for key in range(capacity):
        c.put(key, key)
        ref.put(key, key)
    for _ in range(300_000):
        if rng.random() < 0.5:
            key = rng.randrange(0, 2 * capacity)
            assert c.get(key) == ref.get(key)
        else:
            key = rng.randrange(0, 2 * capacity)
            c.put(key, -key)
            ref.put(key, -key)
