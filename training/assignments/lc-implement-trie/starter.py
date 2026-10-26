"""Implement Trie. Keep the class and method names; the tests import Trie."""


class Trie:
    def __init__(self):
        """Create an empty trie. Each instance must own its own storage."""
        raise NotImplementedError

    def insert(self, word: str) -> None:
        """Store word."""
        raise NotImplementedError

    def search(self, word: str) -> bool:
        """Return True if exactly word was inserted before."""
        raise NotImplementedError

    def starts_with(self, prefix: str) -> bool:
        """Return True if some inserted word starts with prefix."""
        raise NotImplementedError
