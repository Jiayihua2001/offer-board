def accounts_merge(accounts: list[list[str]]) -> list[list[str]]:
    parent, size, owner = {}, {}, {}             # union-find over emails; owner maps email -> name

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:                 # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:                  # union by size
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for name, *emails in accounts:
        for email in emails:
            if email not in parent:
                parent[email], size[email], owner[email] = email, 1, name
            union(emails[0], email)

    groups = {}
    for email in parent:
        groups.setdefault(find(email), []).append(email)
    return [[owner[root]] + sorted(emails) for root, emails in groups.items()]
