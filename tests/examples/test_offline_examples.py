"""Every checked-in example still imports and runs, with no key on the machine.

An example is the first thing a coding agent copies, so a broken one is worse
than a missing one: the agent reads it, pastes it, and gets a traceback it has
no context for. These checks are deliberately cheap -- a ``--help`` that exits 0
proves the module imported, which is where examples actually rot (a helper file
that was never committed, a renamed parameter, a moved import path).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"

# Scripts that parse arguments. ``--help`` runs the whole module body, so this
# catches import-time breakage without paying for a real run.
CLI_EXAMPLES = [
    "agent-behavior/run.py",
    "identity/generate.py",
    "prime-intellect-rl/diagnose.py",
    "prime-intellect-rl/export_prompts.py",
    "prime-intellect-rl/generate.py",
    "schema/migrate.py",
    "schema/project.py",
]

# Needs the `modal` client, which is not a dev dependency. Compiled, not run.
NEEDS_MODAL = {"identity/eval_modal.py", "identity/train_modal.py"}


def _offline_env() -> dict[str, str]:
    """The environment of an agent that has not configured anything yet."""
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "ZEROPROOF_API_KEY", "VLLM_API_KEY",
                "ZEROPROOF_MODEL_URL", "ZEROPROOF_API_URL"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    return env


def _run(script: Path, *args: str, cwd: Path, timeout: int = 120):
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, cwd=str(cwd),
                          env=_offline_env(), timeout=timeout)


def test_every_example_directory_is_tracked_by_git():
    """`.gitignore` has `examples/*`; a new example is invisible until unignored."""
    tracked = subprocess.run(["git", "ls-files", "examples"], capture_output=True,
                             text=True, cwd=str(REPO), check=True).stdout.split()
    tracked_dirs = {Path(p).parts[1] for p in tracked if len(Path(p).parts) > 1}
    on_disk = {d.name for d in EXAMPLES.iterdir()
               if d.is_dir() and not d.name.startswith((".", "__"))}
    assert on_disk - tracked_dirs == set(), (
        "example directory on disk but not in git; add a `!examples/<name>/` "
        "line to .gitignore")


def test_every_example_module_compiles():
    """A syntax error in an example is a broken example, modal client or not."""
    import py_compile

    scripts = sorted(EXAMPLES.glob("*/*.py"))
    assert scripts, "no example scripts found"
    for script in scripts:
        py_compile.compile(str(script), doraise=True)


@pytest.mark.parametrize("rel", CLI_EXAMPLES)
def test_cli_example_imports_and_answers_help(rel, tmp_path):
    """`--help` exits 0, which means every import in the module resolved."""
    out = _run(EXAMPLES / rel, "--help", cwd=tmp_path)
    assert out.returncode == 0, f"{rel} --help failed:\n{out.stderr[-2000:]}"
    assert "usage:" in out.stdout


def test_agent_behavior_selftest_passes_with_no_key(tmp_path):
    """The example's own invariant suite. No model, no network, exit 0."""
    out = _run(EXAMPLES / "agent-behavior/selftest.py", cwd=tmp_path, timeout=300)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
    assert "all checks passed" in out.stdout


def test_agent_behavior_task_pack_is_self_contained():
    """The seven documented tasks load without the optional `tasks_hard` pack."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); import tasks;"
         " print(len(tasks.TASKS), sorted(tasks.BY_ID))"],
        capture_output=True, text=True, cwd=str(EXAMPLES / "agent-behavior"),
        env=_offline_env(), timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    count = int(out.stdout.split()[0])
    assert count >= 7, out.stdout


def test_schema_migrate_runs_end_to_end_offline(tmp_path):
    """Simulates 24 rows and migrates them with no key set."""
    out = _run(EXAMPLES / "schema/migrate.py", "--out", str(tmp_path / "out"),
               cwd=tmp_path, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    assert (tmp_path / "out" / "rows.v1.jsonl").exists()
    assert (tmp_path / "out" / "tasks.jsonl").exists()


def test_example_that_needs_a_key_says_which_one(tmp_path):
    """Fail fast, name the env var, link to where the key comes from."""
    out = _run(EXAMPLES / "agent-behavior/run.py", "--runs", "1", cwd=tmp_path)
    assert out.returncode != 0
    message = out.stdout + out.stderr
    assert "ZEROPROOF_API_KEY" in message, message[-2000:]
    assert "http" in message, message[-2000:]


def test_modal_examples_are_listed_not_forgotten():
    """If a modal example moves, this list is stale and the skip is a lie."""
    for rel in NEEDS_MODAL:
        assert (EXAMPLES / rel).exists(), f"{rel} is listed here but gone from disk"
