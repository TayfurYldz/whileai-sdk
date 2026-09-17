"""Deprecated import path: ``zeroproof_simulations`` is ``whileai.simulations``."""

from __future__ import annotations

import importlib
import sys

import zeroproof  # noqa: F401  installs the alias finder and warns once

sys.modules[__name__] = importlib.import_module("whileai.simulations")
