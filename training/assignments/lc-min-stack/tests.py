import random

from solution import MinStack


def test_example():
    s = MinStack()
    s.push(-2)
    s.push(0)
    s.push(-3)
    assert s.get_min() == -3
    s.pop()
    assert s.top() == 0
    assert s.get_min() == -2


def test_single_element():
    s = MinStack()
    s.push(7)
    assert s.top() == 7
    assert s.get_min() == 7
    s.pop()
    s.push(-1)
    assert s.top() == -1
    assert s.get_min() == -1


def test_duplicate_minimum_survives_one_pop():
    s = MinStack()
    s.push(2)
    s.push(2)
    s.push(5)
    s.pop()
    assert s.get_min() == 2
    s.pop()
    assert s.get_min() == 2, "the other copy of 2 is still on the stack"
    assert s.top() == 2


def test_minimum_reverts_as_elements_pop():
    s = MinStack()
    expected_mins = []
    for x in [5, 3, 7, 3, 1, 9]:
        s.push(x)
        expected_mins.append(min(expected_mins[-1], x) if expected_mins else x)
        assert s.get_min() == expected_mins[-1]
    for _ in range(5):
        s.pop()
        expected_mins.pop()
        assert s.get_min() == expected_mins[-1]
        assert s.top() is not None


def test_increasing_pushes_keep_the_first_minimum():
    s = MinStack()
    for x in range(10, 20):
        s.push(x)
        assert s.get_min() == 10
        assert s.top() == x


def test_random_against_plain_list():
    rng = random.Random(101)
    s = MinStack()
    ref = []
    for _ in range(3000):
        if not ref or rng.random() < 0.6:
            x = rng.randrange(-20, 21)
            s.push(x)
            ref.append(x)
        else:
            s.pop()
            ref.pop()
        if ref:
            assert s.top() == ref[-1]
            assert s.get_min() == min(ref)


def test_large_number_of_operations_is_fast():
    s = MinStack()
    n = 200_000
    for x in range(n, 0, -1):
        s.push(x)
        assert s.get_min() == x
    for x in range(1, n + 1):
        assert s.get_min() == x
        assert s.top() == x
        s.pop()
