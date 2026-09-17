"""Every paper recipe under ``recipes/papers/`` keeps the contract, its table
row is current, and it still compiles."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "recipes" / "papers"


def test_papers_check_passes() -> None:
    out = subprocess.run(
        [sys.executable, str(PAPERS / "check.py")], capture_output=True, text=True, cwd=REPO
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_every_paper_recipe_compiles() -> None:
    scripts = sorted(PAPERS.glob("*/recipe.py"))
    assert scripts, "no recipe.py under recipes/papers/ (the template counts)"
    for script in scripts:
        py_compile.compile(str(script), doraise=True)
