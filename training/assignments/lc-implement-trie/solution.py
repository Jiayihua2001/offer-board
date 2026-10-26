class _Node:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}                   # char -> _Node
        self.is_end = False                  # some inserted word ends exactly here


class Trie:
    def __init__(self):
        self.root = _Node()                  # per instance, never a class attribute

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = node.children[ch] = _Node()
            node = nxt
        node.is_end = True

    def _walk(self, s: str):
        """The node reached by following s, or None if the path breaks off."""
        node = self.root
        for ch in s:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        return self._walk(prefix) is not None
