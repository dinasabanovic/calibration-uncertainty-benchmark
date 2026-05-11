"""
utils.py  —  shared constants and helpers
"""
import warnings
from typing import Any

EPS: float = 1e-8


def deprecated(msg: str):
    def decorator(fn):
        def wrapper(*args, **kwargs) -> Any:
            warnings.warn(f"{fn.__name__} is deprecated. {msg}",
                          DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__  = fn.__doc__
        return wrapper
    return decorator
