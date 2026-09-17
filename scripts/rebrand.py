#!/usr/bin/env python3
"""Rename the ``zeroproof`` package to ``whileai`` (ZeroProof -> While).

This is a regenerable change, not a hand-made diff. Run it on a clean
checkout of ``main`` and commit the result; run it on an open branch instead
of resolving rename conflicts by hand. It is meant to run exactly once per
branch and refuses to run twice (there is no ``zeroproof/`` directory left
afterwards).

    python scripts/rebrand.py && uv lock && uv sync --extra dev
    python scripts/rebrand.py --alias   # on a branch renamed before zps became wai
    uv run ruff check --fix --select I,F401 . && uv run ruff format .

What it does, in order:

1. ``git mv zeroproof whileai``, drop the old ``zeroproof_simulations``
   alias package, move ``skills/zeroproof-simulations``.
2. Rewrite every tracked text file: module paths, the import name, the CLI
   name, ``ZEROPROOF_*`` -> ``WHILEAI_*``, the brand words, the GitHub org.
   Left alone on purpose: hosts (``api.zeroproofai.com``, the Modal
   ``*.modal.run`` apps), the Hugging Face org, span attribute keys such as
   ``zeroproof.model_version`` (a platform contract), ``uv.lock`` (regenerate
   it), and ``CHANGELOG.md`` history (an entry is added at the top instead).
3. Route every environment read through ``whileai._env.getenv`` so the old
   ``ZEROPROOF_*`` names keep working, and let ``~/.zeroproof`` credentials
   be found from the new default ``~/.whileai``.
4. Write ``compat/zeroproof``: the final releases of the old PyPI name are a
   shim that depends on ``whileai`` and aliases ``import zeroproof`` (and
   ``zeroproof_simulations``) to the same module objects.
5. Point the publish workflow at both distributions and let the version gate
   continue the numbering from the old name.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_FILES = {"uv.lock", "CHANGELOG.md", "scripts/rebrand.py"}
# Lines that name infrastructure which did not move. Left byte-for-byte.
PROTECT_LINE = ("modal.run", "zeroproofai--", '"zeroproof" in')
# Span attribute keys are read by the platform; ``zeroproof.`` followed by a
# non-module segment (or by punctuation, as in prose about the prefix) stays.
SPAN_KEYS = r"scenario_id|scores|describe|evidence|persona|task|model_version"
# Modal app names (zeroproof-judge, zeroproof-serve-*, ...) are hosts too.
MODAL_APPS = r"judge|serve|embed|studio"
SPAN = re.compile(rf"zeroproof(?=\.(?:(?:{SPAN_KEYS})\b|[^\w])|-(?:{MODAL_APPS})\b)")
PLACEHOLDER = "\x00SPAN\x00"
# The conventional import alias. ``zps`` spelled out the old name; ``wai``
# is While AI. Also covers helpers named after it (``zps_write``).
ALIAS = re.compile(r"\bzps(?![A-Za-z0-9])")


def sh(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, cwd=ROOT).stdout


def fix_line(line: str) -> str:
    if any(p in line for p in PROTECT_LINE):
        return line
    line = SPAN.sub(PLACEHOLDER, line)
    line = line.replace("zeroproof_simulations", "whileai.simulations")
    line = re.sub(r"\bzeroproof\b", "whileai", line)
    line = line.replace("ZEROPROOF_", "WHILEAI_")
    line = line.replace("Zero-Proof-AI", "whilehq")
    line = re.sub(r"Zero Proof (?:AI|Labs)", "While", line)
    line = line.replace("Zero Proof", "While").replace("ZeroProof", "While")
    line = ALIAS.sub("wai", line)
    return line.replace(PLACEHOLDER, "zeroproof")


def rewrite_tree(fix=fix_line) -> int:
    changed = 0
    for rel in sh("git", "ls-files").split("\n"):
        if not rel or rel in SKIP_FILES:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8")
        new = "".join(fix(ln) for ln in text.splitlines(keepends=True))
        if new != text:
            path.write_bytes(new.encode("utf-8"))
            changed += 1
    return changed


def must_replace(rel: str, old: str, new: str, count: int = 1) -> None:
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        sys.exit(f"{rel}: expected {count} occurrence(s) of {old!r}, found {found}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# environment reads


ENV_HELPER = '''"""Environment variables: ``WHILEAI_*`` first, then the ``ZEROPROOF_*`` name from before the rename."""

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
'''

ENV_READ = re.compile(r'os\.environ\.get\(\s*"WHILEAI_([A-Z_]+)"')


def route_env_reads() -> None:
    write("whileai/_env.py", ENV_HELPER)
    for path in (ROOT / "whileai").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new = ENV_READ.sub(r'getenv("\1"', text)
        if new == text:
            continue
        if "from whileai._env import getenv" not in new:
            # After ``import os`` (every one of these files has it); isort
            # moves it into the first-party block afterwards.
            new, n = re.subn(
                r"^import os\n",
                "import os\n\nfrom whileai._env import getenv\n",
                new,
                count=1,
                flags=re.M,
            )
            if n != 1:
                sys.exit(f"{path}: could not place the getenv import")
        path.write_text(new, encoding="utf-8", newline="\n")


def patch_auth() -> None:
    must_replace(
        "whileai/auth.py",
        'def config_dir() -> Path:\n    """``$WHILEAI_HOME`` or ``~/.whileai``."""\n'
        '    return Path(getenv("HOME") or Path.home() / ".whileai")\n',
        "def config_dir() -> Path:\n"
        '    """``$WHILEAI_HOME`` (or the older ``$ZEROPROOF_HOME``), else ``~/.whileai``.\n\n'
        "    A ``~/.zeroproof`` left by the package's old name is used as long as\n"
        "    ``~/.whileai`` holds no credentials, so an existing login survives\n"
        "    the rename without signing in again.\n"
        '    """\n'
        '    override = getenv("HOME")\n'
        "    if override:\n"
        "        return Path(override)\n"
        '    new = Path.home() / ".whileai"\n'
        '    old = Path.home() / ".zeroproof"\n'
        '    if not (new / "credentials.json").exists() and (old / "credentials.json").exists():\n'
        "        return old\n"
        "    return new\n",
    )
    must_replace(
        "whileai/auth.py",
        '    env = getenv("API_KEY")\n',
        '    env = getenv("API_KEY")\n    env_var = env_name("API_KEY")\n',
    )
    must_replace(
        "whileai/auth.py",
        '"source": "WHILEAI_API_KEY" if env else ("file" if saved.get("api_key") else None),',
        '"source": env_var if env else ("file" if saved.get("api_key") else None),',
    )
    must_replace(
        "whileai/auth.py",
        "from whileai._env import getenv\n",
        "from whileai._env import env_name, getenv\n",
    )
    # reference.py keeps its host check line verbatim (protected); the env
    # read on that line still has to honor both names.
    must_replace(
        "whileai/simulations/score/reference.py",
        'os.environ.get("ZEROPROOF_API_KEY", "").strip() if "zeroproof" in base_url else ""',
        'getenv("API_KEY", "").strip() if "zeroproof" in base_url else ""',
    )
    must_replace(
        "whileai/simulations/score/reference.py",
        "import os\n",
        "import os\n\nfrom whileai._env import getenv\n",
    )
    # A generated environment's fallback name.
    must_replace(
        "whileai/simulations/environment.py",
        'or "zeroproof_env"',
        'or "whileai_env"',
    )


def patch_ingest_alias() -> None:
    must_replace(
        "whileai/ingest.py",
        "class WhileIngestError(Exception):",
        "class WhileIngestError(Exception):",
    )
    path = ROOT / "whileai/ingest.py"
    path.write_text(
        path.read_text(encoding="utf-8").rstrip("\n")
        + "\n\n\n# The name this exception had before the package was renamed.\n"
        "ZeroProofIngestError = WhileIngestError\n",
        encoding="utf-8",
        newline="\n",
    )
    must_replace(
        "whileai/__init__.py",
        "from .ingest import (\n    WhileIngestError,\n",
        "from .ingest import (\n    WhileIngestError,\n    ZeroProofIngestError,\n",
    )
    must_replace(
        "whileai/__init__.py",
        '    "WhileIngestError",\n',
        '    "WhileIngestError",\n    "ZeroProofIngestError",\n',
    )


def patch_conftest() -> None:
    must_replace(
        "tests/conftest.py",
        '    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)\n',
        '    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)\n'
        '    monkeypatch.delenv("ZEROPROOF_API_KEY", raising=False)\n',
    )
    must_replace(
        "tests/conftest.py",
        '    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path / "whileai-home"))\n',
        '    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path / "whileai-home"))\n'
        '    monkeypatch.delenv("ZEROPROOF_HOME", raising=False)\n',
    )


# --------------------------------------------------------------------------
# packaging


def patch_pyproject() -> None:
    must_replace(
        "pyproject.toml",
        'include = ["whileai*", "whileai.simulations"]',
        'include = ["whileai*"]',
    )
    must_replace(
        "pyproject.toml",
        'known-first-party = ["whileai", "whileai.simulations", "tests"]',
        'known-first-party = ["whileai", "tests"]',
    )
    must_replace(
        "MANIFEST.in",
        "recursive-include whileai.simulations *.py\nrecursive-include whileai *.py\n"
        "recursive-include whileai.simulations *.py\n",
        "recursive-include whileai *.py\n",
    )
    must_replace(
        ".gitignore",
        "!/scripts/golden.py\n",
        "!/scripts/golden.py\n# ...and the rename script, so an open branch can regenerate the rebrand.\n!/scripts/rebrand.py\n",
    )


COMPAT_PYPROJECT = """[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

# The old name of this SDK. Each release here is a shim that installs
# ``whileai`` and keeps ``import zeroproof`` working. The version and the
# ``whileai>=`` floor must equal the version in ../../pyproject.toml; the
# publish gate (.github/scripts/check_version.py) checks that.
[project]
name = "zeroproof"
version = "{version}"
description = "Renamed: the ZeroProof SDK is now whileai. Installs whileai and keeps `import zeroproof` working."
readme = "README.md"
license = {{text = "Apache-2.0"}}
requires-python = ">=3.10"
dependencies = ["whileai>={version}"]
authors = [{{name = "While"}}]
classifiers = [
    "Development Status :: 7 - Inactive",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
]

[project.urls]
Homepage = "https://github.com/whilehq/whileai-sdk"
Repository = "https://github.com/whilehq/whileai-sdk"
Issues = "https://github.com/whilehq/whileai-sdk/issues"

[project.scripts]
zeroproof = "whileai.cli:main"
zeroproof-simulations = "whileai.simulations.score.quality:main"

[tool.setuptools]
packages = ["zeroproof", "zeroproof_simulations"]
include-package-data = false
"""

COMPAT_README = """# zeroproof

The ZeroProof SDK was renamed **whileai** (ZeroProof is now [While](https://while.ai)).

```bash
pip install whileai
```

```python
import whileai
import whileai.simulations as wai
```

This package is the old name. Installing it installs `whileai` and keeps
`import zeroproof` (and the older `import zeroproof_simulations`) resolving to
the very same modules, with a `DeprecationWarning`. The `zeroproof` command
still runs the CLI. `ZEROPROOF_*` environment variables and a saved
`~/.zeroproof/credentials.json` are still read by `whileai`.

Change the import when you can; this shim is not where new releases land.
Source: https://github.com/whilehq/whileai-sdk (`compat/zeroproof`).
"""

COMPAT_INIT = '''"""``zeroproof`` was renamed ``whileai``. This package keeps the old import working.

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
'''

COMPAT_SIMS_INIT = '''"""Deprecated import path: ``zeroproof_simulations`` is ``whileai.simulations``."""

from __future__ import annotations

import importlib
import sys

import zeroproof  # noqa: F401  installs the alias finder and warns once

sys.modules[__name__] = importlib.import_module("whileai.simulations")
'''

COMPAT_TEST = '''"""``compat/zeroproof``: the old import names are aliases of ``whileai``, not copies.

The shim is a separate distribution and is not installed in the dev
environment; the test puts its source on ``sys.path`` instead.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest

COMPAT = Path(__file__).resolve().parents[2] / "compat" / "zeroproof"


@pytest.fixture
def compat_path(monkeypatch):
    monkeypatch.syspath_prepend(str(COMPAT))
    for name in list(sys.modules):
        if name == "zeroproof" or name.startswith(("zeroproof.", "zeroproof_simulations")):
            monkeypatch.delitem(sys.modules, name)
    yield
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_AliasFinder"]


def test_zeroproof_is_whileai(compat_path):
    import whileai
    import whileai.simulations as new

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = importlib.import_module("zeroproof")
    assert old is whileai
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert importlib.import_module("zeroproof.simulations") is new
    assert new.__spec__.name == "whileai.simulations"  # alias must not clobber it
    assert (
        importlib.import_module("zeroproof.simulations.run.engine")
        is importlib.import_module("whileai.simulations.run.engine")
    )
    assert old.ZeroProofIngestError is whileai.WhileIngestError


def test_zeroproof_simulations_is_whileai_simulations(compat_path):
    import whileai.simulations as new

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = importlib.import_module("zeroproof_simulations")
        judging = importlib.import_module("zeroproof_simulations.score.judging")
    assert legacy is new
    assert judging is importlib.import_module("whileai.simulations.score.judging")
'''


def write_compat(version: str) -> None:
    write("compat/zeroproof/pyproject.toml", COMPAT_PYPROJECT.format(version=version))
    write("compat/zeroproof/README.md", COMPAT_README)
    write("compat/zeroproof/zeroproof/__init__.py", COMPAT_INIT)
    write("compat/zeroproof/zeroproof_simulations/__init__.py", COMPAT_SIMS_INIT)
    write("tests/api/test_legacy_import_path.py", COMPAT_TEST)


# --------------------------------------------------------------------------
# workflows


def patch_workflows() -> None:
    must_replace(
        ".github/workflows/publish.yml",
        "      - run: uv build\n      - run: uvx twine check dist/*\n",
        "      - run: uv build --out-dir dist\n"
        "      # The old name: a shim that depends on this release. Same version,\n"
        "      # same run, so `pip install zeroproof` never lags `pip install whileai`.\n"
        "      - run: uv build --out-dir dist compat/zeroproof\n"
        "      - run: uvx twine check dist/*\n",
    )
    must_replace(
        ".github/workflows/publish.yml",
        '          if [ -n "$UV_PUBLISH_TOKEN" ]; then\n'
        "            uv publish --check-url https://pypi.org/simple/whileai/\n"
        "          else\n"
        "            uv publish --trusted-publishing always --check-url https://pypi.org/simple/whileai/\n"
        "          fi\n",
        "          # Both projects on PyPI must accept this workflow: either the\n"
        "          # token covers both, or each project lists it as a trusted\n"
        "          # publisher (owner whilehq, repo whileai-sdk, publish.yml, env pypi).\n"
        "          for name in whileai zeroproof; do\n"
        '            if [ -n "$UV_PUBLISH_TOKEN" ]; then\n'
        '              uv publish --check-url "https://pypi.org/simple/$name/" dist/$name-*\n'
        "            else\n"
        '              uv publish --trusted-publishing always --check-url "https://pypi.org/simple/$name/" dist/$name-*\n'
        "            fi\n"
        "          done\n",
    )
    must_replace(
        ".github/workflows/ci.yml",
        "      - run: python -m build\n      - run: python -m twine check dist/*\n",
        "      - run: python -m build\n"
        "      - run: python -m build --outdir dist compat/zeroproof\n"
        "      - run: python -m twine check dist/*\n",
    )
    must_replace(
        ".github/workflows/ci.yml",
        "          /tmp/fresh/bin/pip install --quiet dist/*.whl\n",
        "          /tmp/fresh/bin/pip install --quiet dist/whileai-*.whl\n"
        "          /tmp/fresh/bin/pip install --quiet --no-deps dist/zeroproof-*.whl\n",
    )
    must_replace(
        ".github/workflows/ci.yml",
        "          import whileai.simulations as wai\n"
        "          import whileai.simulations as legacy\n"
        "          assert legacy is zps, 'compatibility shim must alias the real package'\n",
        "          import whileai.simulations as wai\n"
        "          import zeroproof.simulations as old\n"
        "          import zeroproof_simulations as legacy\n"
        "          assert old is wai and legacy is wai, 'compatibility shim must alias the real package'\n",
    )


def patch_version_gate() -> None:
    must_replace(
        ".github/scripts/check_version.py",
        'PYPI = "https://pypi.org/pypi/{name}/json"\n',
        'PYPI = "https://pypi.org/pypi/{name}/json"\n'
        "# The old name of this package. Its releases count as prior releases of\n"
        "# the new name (the numbering continues across the rename), and every\n"
        "# release ships a shim under it that must carry the same version.\n"
        'COMPAT = "compat/zeroproof/pyproject.toml"\n',
    )
    must_replace(
        ".github/scripts/check_version.py",
        "    out = []\n"
        '    for raw in data.get("releases", {}):\n'
        "        try:\n"
        "            out.append(Version(raw).release)\n"
        "        except InvalidVersion:\n"
        "            continue\n"
        "    return sorted(out)\n",
        "    out = []\n"
        '    for raw in data.get("releases", {}):\n'
        "        try:\n"
        "            release = Version(raw).release\n"
        "        except InvalidVersion:\n"
        "            continue\n"
        "        # A name-reservation upload (0.0.1) is not part of the scheme.\n"
        "        if len(release) == 2:\n"
        "            out.append(release)\n"
        "    return sorted(out)\n"
        "\n"
        "\n"
        "def compat_check(name: str, version: str) -> tuple[str, list[tuple[int, ...]]]:\n"
        '    """The shim\'s name and published releases; fail if it is out of step."""\n'
        "    old_name, old_version = local_version(COMPAT)\n"
        "    if old_version != version:\n"
        '        fail(f"{COMPAT} is at {old_version}, pyproject.toml is at {version}; keep them equal")\n'
        '    with open(COMPAT, "rb") as fh:\n'
        '        deps = tomllib.load(fh)["project"]["dependencies"]\n'
        '    if f"{name}>={version}" not in deps:\n'
        '        fail(f"{COMPAT} must depend on {name}>={version}, has {deps}")\n'
        "    return old_name, published(old_name)\n",
    )
    must_replace(
        ".github/scripts/check_version.py",
        '    prior = published(name)\n    print(f"package        : {name}")\n',
        "    prior = published(name)\n"
        "    old_name, old_prior = compat_check(name, version)\n"
        "    if not prior:\n"
        "        prior = old_prior  # continue the numbering from the old name\n"
        '    print(f"package        : {name} (was {old_name})")\n',
    )


# --------------------------------------------------------------------------
# docs


def patch_docs(version: str) -> None:
    must_replace(
        "README.md",
        "- `whileai.simulations`: post-training data for an agent. (Was the separate top-level "
        "package `whileai.simulations`; that name still imports for two releases with a "
        "deprecation warning.) Give it",
        "- `whileai.simulations`: post-training data for an agent. Give it",
    )
    must_replace(
        "README.md",
        "This repo absorbed the `whileai-simulations` package; `whileai-simulations` on PyPI "
        "is deprecated in favor of `whileai`.\n\n"
        "Releases of `whileai` before 0.3 were an unrelated encrypted agent-to-agent messaging "
        "client. That code was removed in 0.04; pin `whileai<0.3` if you still depend on it.\n",
        "**Renamed.** This SDK was `zeroproof` (ZeroProof is now While). `pip install zeroproof` "
        "still works: it installs `whileai`, and `import zeroproof` (or the older "
        "`zeroproof_simulations`) resolves to the same modules with a deprecation warning. "
        "`ZEROPROOF_*` environment variables and a saved `~/.zeroproof/credentials.json` are "
        "still read. Change the import when you can; new releases land under `whileai`.\n\n"
        "Releases of `zeroproof` before 0.3 were an unrelated encrypted agent-to-agent messaging "
        "client. That code was removed in 0.04; pin `zeroproof<0.3` if you still depend on it.\n",
    )
    must_replace(
        "RELEASING.md",
        "## Cutting a release\n",
        "## Two distributions, one version\n\n"
        "`compat/zeroproof` is the package's old name: a shim that depends on `whileai` and\n"
        "aliases `import zeroproof` to it. It is built and uploaded by the same publish\n"
        "run, so both `pyproject.toml` files carry the same version and the shim's\n"
        "`whileai>=` floor equals it. The gate fails a release where they differ. Bump\n"
        "both files together.\n\n"
        "## Cutting a release\n",
    )
    changelog = ROOT / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8")
    entry = (
        "- **Renamed to `whileai`.** ZeroProof is now While, and the package follows:\n"
        "  `pip install whileai`, `import whileai`, `import whileai.simulations as wai`,\n"
        "  the `whileai` command, `WHILEAI_*` environment variables, `~/.whileai` for\n"
        "  the saved login, and `whileai.WhileIngestError`. Nothing old breaks: the\n"
        "  `zeroproof` distribution keeps releasing as a shim (`compat/zeroproof`) that\n"
        "  installs `whileai` and aliases `import zeroproof` and\n"
        "  `import zeroproof_simulations` to the same module objects with a\n"
        "  `DeprecationWarning`; the `zeroproof` command still runs; every\n"
        "  `ZEROPROOF_*` variable is read when its `WHILEAI_*` twin is unset; a\n"
        "  `~/.zeroproof/credentials.json` is used until `~/.whileai` has one;\n"
        "  `ZeroProofIngestError` is an alias of `WhileIngestError`. Hosts\n"
        "  (`api.zeroproofai.com`, the Modal apps), the `zp_` key prefix, the\n"
        "  Hugging Face org and the `zeroproof.*` span attributes are unchanged. The\n"
        "  repository moved to `whilehq/whileai-sdk`. The rename is `scripts/rebrand.py`,\n"
        "  a script to run on an open branch instead of resolving conflicts by hand.\n"
    )
    if "\n## Unreleased\n" in text:
        # Another change is already queued for the next release; lead its list.
        head, rest = text.split("\n## Unreleased\n", 1)
        text = head + "\n## Unreleased\n\n" + entry + rest.lstrip("\n")
    else:
        head, rest = text.split("\n## ", 1)
        text = head + "\n## Unreleased\n\n" + entry + "\n## " + rest
    changelog.write_text(text, encoding="utf-8", newline="\n")
    _ = version


# --------------------------------------------------------------------------


def main() -> int:
    if "--alias" in sys.argv[1:]:
        # For a branch that renamed before the alias changed: only the
        # ``zps`` -> ``wai`` step, idempotent.
        changed = rewrite_tree(lambda ln: ALIAS.sub("wai", ln))
        print(f"alias: rewrote {changed} files")
        return 0
    if not (ROOT / "zeroproof").is_dir():
        sys.exit("no zeroproof/ directory: the rename already ran on this branch")
    if sh("git", "status", "--porcelain").strip():
        sys.exit("working tree is not clean; commit or discard first")
    import tomllib

    with open(ROOT / "pyproject.toml", "rb") as fh:
        version = tomllib.load(fh)["project"]["version"]

    sh("git", "clean", "-fdxq", "zeroproof", "zeroproof_simulations")
    sh("git", "mv", "zeroproof", "whileai")
    sh("git", "rm", "-r", "-q", "zeroproof_simulations")
    sh("git", "mv", "skills/zeroproof-simulations", "skills/whileai-simulations")
    changed = rewrite_tree()
    route_env_reads()
    patch_auth()
    patch_ingest_alias()
    patch_conftest()
    patch_pyproject()
    write_compat(version)
    patch_workflows()
    patch_version_gate()
    patch_docs(version)
    sh("git", "add", "-A", ".")
    print(f"rewrote {changed} files; version {version}; now: uv lock && uv sync --extra dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
