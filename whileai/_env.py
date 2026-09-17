"""Environment variables: ``WHILEAI_*`` first, then the ``ZEROPROOF_*`` name from before the rename."""

from __future__ import annotations

import os
from typing import overload

NEW_PREFIX = "WHILEAI_"
OLD_PREFIX = "ZEROPROOF_"


@overload
def getenv(name: str) -> str | None: ...
@overload
def getenv(name: str, default: str) -> str: ...


def getenv(name: str, default: str | None = None) -> str | None:
    """Read ``WHILEAI_<name>``, else ``ZEROPROOF_<name>``, else ``default``.

    An empty string counts as unset, which is how every caller treated the
    old variables (``os.environ.get(...) or fallback``).
    """
    for prefix in (NEW_PREFIX, OLD_PREFIX):
        value = os.environ.get(prefix + name)
        if value:
            return value
    return default


def env_name(name: str) -> str | None:
    """Which variable ``getenv(name)`` would read, or ``None`` if neither is set."""
    for prefix in (NEW_PREFIX, OLD_PREFIX):
        if os.environ.get(prefix + name):
            return prefix + name
    return None
