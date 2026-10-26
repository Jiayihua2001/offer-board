"""LRU Cache. Keep the class and method names; the tests import the class."""


class LRUCache:
    def __init__(self, capacity: int) -> None:
        raise NotImplementedError

    def get(self, key: int) -> int:
        """Return the value for key and mark it most recently used, or -1 if absent."""
        raise NotImplementedError

    def put(self, key: int, value: int) -> None:
        """Insert or update key, mark it most recently used, evicting the least recently used if needed."""
        raise NotImplementedError
