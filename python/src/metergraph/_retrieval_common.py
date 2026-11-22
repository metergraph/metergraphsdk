"""Shared accessors for native retrieval usage."""
from collections.abc import Mapping

def get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def count(value):
    return value if type(value) is int and value >= 0 else None
