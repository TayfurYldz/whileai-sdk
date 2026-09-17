"""``zeroproof`` was renamed ``whileai``. This package keeps the old import working.

``import zeroproof`` and every submodule path under it resolve to the very same
module objects as ``whileai``, so ``zeroproof.simulations.run.engine is
whileai.simulations.run.engine``. The older ``zeroproof_simulations`` name is
aliased to ``whileai.simulations`` the same way.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

ALIASES = {"zeroproof": "whileai", "zeroproof_simulations": "whileai.simulations"}


def _target(name: str) -> str | None:
    for old, new in ALIASES.items():
        if name == old or name.startswith(old + "."):
            return new + name[len(old) :]
    return None


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve ``zeroproof.x`` from ``whileai.x``."""

    def find_spec(self, name, path=None, target=None):
        if _target(name) is None:
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        real = _target(spec.name)
        assert real is not None
        module = importlib.import_module(real)
        # The import system stamps this alias spec onto the module before
        # exec_module; put the real one back so resource lookups on the
        # whileai name keep working.
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self._real_spec


warnings.warn(
    "The zeroproof package was renamed whileai. Run `pip install whileai` and change "
    "`import zeroproof` to `import whileai`; the old name is a shim and stops "
    "receiving releases.",
    DeprecationWarning,
    stacklevel=2,
)

if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
sys.modules[__name__] = importlib.import_module("whileai")
