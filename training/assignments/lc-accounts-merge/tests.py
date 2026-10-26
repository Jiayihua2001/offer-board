import random
from collections import deque

from solution import accounts_merge


def brute_force(accounts):
    """Independent answer: BFS over accounts, where two accounts are adjacent if they share an email."""
    owners = {}
    for i, (_, *emails) in enumerate(accounts):
        for e in emails:
            owners.setdefault(e, []).append(i)
    seen, out = set(), []
    for i in range(len(accounts)):
        if i in seen:
            continue
        seen.add(i)
        queue, emails = deque([i]), set()
        while queue:
            j = queue.popleft()
            for e in accounts[j][1:]:
                emails.add(e)
                for k in owners[e]:
                    if k not in seen:
                        seen.add(k)
                        queue.append(k)
        out.append([accounts[i][0]] + sorted(emails))
    return out


def normalise(merged):
    return sorted(tuple(acc) for acc in merged)


def check(accounts):
    got = accounts_merge([list(acc) for acc in accounts])
    assert isinstance(got, list), f"expected a list, got {type(got).__name__}"
    for acc in got:
        emails = acc[1:]
        assert emails == sorted(emails), f"emails are not sorted in {acc}"
        assert len(set(emails)) == len(emails), f"duplicate emails in {acc}"
    want = brute_force(accounts)
    assert normalise(got) == normalise(want), f"expected {normalise(want)}, got {normalise(got)}"
    return got


def test_example():
    accounts = [
        ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
        ["John", "johnsmith@mail.com", "john00@mail.com"],
        ["Mary", "mary@mail.com"],
        ["John", "johnnybravo@mail.com"],
    ]
    got = check(accounts)
    assert len(got) == 3
    assert ["John", "john00@mail.com", "john_newyork@mail.com", "johnsmith@mail.com"] in got


def test_nothing_to_merge_and_single_account():
    assert accounts_merge([["Ann", "a@x.com"]]) == [["Ann", "a@x.com"]]
    got = check([["Ann", "a@x.com"], ["Bob", "b@x.com"], ["Cid", "c@x.com", "d@x.com"]])
    assert len(got) == 3


def test_chain_of_accounts_merges_transitively():
    accounts = [["Zed", "a@x.com", "b@x.com"], ["Zed", "c@x.com", "d@x.com"],
                ["Zed", "b@x.com", "c@x.com"], ["Zed", "d@x.com", "e@x.com"]]
    got = check(accounts)
    assert got == [["Zed", "a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"]]


def test_same_name_is_not_the_same_person():
    got = check([["Lee", "lee1@x.com"], ["Lee", "lee2@x.com"], ["Lee", "lee3@x.com"]])
    assert len(got) == 3


def test_duplicate_emails_within_an_account():
    got = check([["A", "x@m.com"], ["A", "x@m.com", "x@m.com"], ["A", "y@m.com", "y@m.com", "x@m.com"]])
    assert got == [["A", "x@m.com", "y@m.com"]]


def test_random_accounts_match_brute_force():
    rng = random.Random(8)
    for _ in range(150):
        n_people = rng.randrange(1, 6)
        n_accounts = rng.randrange(1, 12)
        accounts = []
        for _ in range(n_accounts):
            person = rng.randrange(n_people)
            k = rng.randrange(1, 5)
            emails = [f"p{person}e{rng.randrange(6)}@m.com" for _ in range(k)]
            accounts.append([f"Name{person}"] + emails)
        check(accounts)


def test_large_input_is_fast():
    rng = random.Random(99)
    n_people, accounts = 4000, []
    for _ in range(20_000):
        person = rng.randrange(n_people)
        k = rng.randrange(1, 5)
        accounts.append([f"N{person}"] + [f"p{person}e{rng.randrange(12)}@m.com" for _ in range(k)])
    got = check(accounts)
    assert len(got) < len(accounts) // 2, "most accounts should have merged"
