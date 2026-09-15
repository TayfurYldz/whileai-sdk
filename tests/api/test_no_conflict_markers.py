"""No tracked file carries a git conflict marker.

A bad merge left ``<<<<<<< HEAD`` in CHANGELOG.md and it shipped on main
(709eb70 removed it). Nothing caught it: the lint job runs ruff and mypy,
which only read Python, so markers in Markdown, JSON or YAML pass CI
untouched. This test reads every tracked text file instead.

The same bad-merge class can leave a control character where a heading
was: 23baeff turned a ``## 0.32 (`` prefix in CHANGELOG.md into a lone
``\x01``, and it survived three days of merges because nothing reads
Markdown for bytes. So the second test rejects C0 control characters
(tab, newline and carriage return excepted) in the same files.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator

from tests.helpers import REPO_ROOT

# Built rather than written literally, so this file does not match itself.
# Only the three unambiguous markers: a bare "=======" is also a Markdown
# setext underline, and every real conflict carries an opener anyway.
MARKERS = tuple(char * 7 for char in "<>|")

# Everything below 0x20 except tab, newline and carriage return.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert out.returncode == 0, out.stderr[-500:]
    return [name for name in out.stdout.split("\0") if name]


def _tracked_text_files() -> Iterator[tuple[str, str]]:
    names = _tracked_files()
    assert len(names) > 50, f"git ls-files returned only {len(names)} paths"
    for name in names:
        path = REPO_ROOT / name
        if not path.is_file():
            continue  # submodule, or a symlink to nowhere
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary; the SVG reads fine, images do not
        yield name, text


def test_no_tracked_file_has_a_conflict_marker():
    offenders: list[str] = []
    scanned = 0
    for name, text in _tracked_text_files():
        scanned += 1
        for number, line in enumerate(text.splitlines(), 1):
            if any(line.startswith(marker) for marker in MARKERS):
                offenders.append(f"{name}:{number}: {line[:70]}")
    assert scanned > 50, f"only {scanned} text files were readable"
    assert not offenders, "conflict markers left in tracked files:\n" + "\n".join(offenders)


def test_no_tracked_text_file_has_a_control_character():
    offenders: list[str] = []
    scanned = 0
    for name, text in _tracked_text_files():
        scanned += 1
        for number, line in enumerate(text.splitlines(), 1):
            hit = _CONTROL.search(line)
            if hit:
                offenders.append(f"{name}:{number}: {hit.group()!r} in {line[:60]!r}")
    assert scanned > 50, f"only {scanned} text files were readable"
    assert not offenders, "control characters in tracked text files:\n" + "\n".join(offenders)
