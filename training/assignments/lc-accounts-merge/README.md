# Accounts Merge

`accounts` is a list where each element is `[name, email_1, email_2, ...]` with at least one email. Two accounts belong to the same person if they share at least one email, and the relation is transitive: if A shares an email with B and B shares one with C, then A, B and C are one person. Two accounts with the same name but no shared email are different people, and all accounts of one person carry the same name.

Return the merged accounts, each as `[name, e_1, e_2, ...]` with the emails sorted in Python's default string order and without duplicates. The merged accounts themselves may be returned in any order.

## Examples

```
accounts_merge([
  ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
  ["John", "johnsmith@mail.com", "john00@mail.com"],
  ["Mary", "mary@mail.com"],
  ["John", "johnnybravo@mail.com"],
])
-> [
  ["John", "john00@mail.com", "john_newyork@mail.com", "johnsmith@mail.com"],
  ["Mary", "mary@mail.com"],
  ["John", "johnnybravo@mail.com"],
]      (the three accounts in any order)

accounts_merge([["A", "x@m.com"], ["A", "x@m.com", "x@m.com"]])  -> [["A", "x@m.com"]]
```

## Constraints

- `1 <= len(accounts) <= 2 * 10^4`, each with 1 to 10 emails; emails are non-empty strings.
- Target: near O(N α(N)) with union-find, or O(N) with DFS over an email graph, where `N` is the total number of emails, plus the sort of each group.

## What the interviewer listens for

- Brute force: compare every pair of accounts for a shared email (O(A^2 * E)) and merge; you say why that is too slow for 2 * 10^4 accounts and why a single pass of pairwise merging is not even enough (chains).
- You pick a graph model (emails as nodes, or accounts as nodes joined through an email -> accounts map) and can explain either union-find or DFS on it.
- Union-find details: path compression, union by size or rank, and the resulting near-constant amortised cost per operation.
- You keep the name attached (an email -> name map or a representative account index) and handle duplicate emails inside one account.

## Follow-up

Accounts now arrive one at a time and after each you must answer "how many distinct people so far?". Which approach adapts, and what does each query cost?
