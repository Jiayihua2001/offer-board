def find_words(board: list[list[str]], words: list[str]) -> list[str]:
    END = "$"                                    # key under which a node stores the word ending there
    root = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node[END] = w

    m, n = len(board), len(board[0])
    found = []

    def dfs(r, c, parent):
        ch = board[r][c]
        node = parent[ch]
        word = node.pop(END, None)               # report once, then forget it
        if word is not None:
            found.append(word)
        board[r][c] = "#"                        # visited marker; "#" is never a trie key
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < m and 0 <= nc < n and board[nr][nc] in node:
                dfs(nr, nc, node)
        board[r][c] = ch
        if not node:
            del parent[ch]                       # prune: nothing below leads to an unfound word any more

    for r in range(m):
        for c in range(n):
            if board[r][c] in root:
                dfs(r, c, root)
    return found
