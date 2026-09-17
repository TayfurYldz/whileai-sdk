"""Every paper recipe has the README sections, a results.json with the right
keys, and a row in the table in this folder's README, which is generated
from the results files so parallel authors never edit the same lines.

    python recipes/papers/check.py          # verify; exit 1 on the first miss
    python recipes/papers/check.py --write  # regenerate the table, then verify
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PAPERS = Path(__file__).resolve().parent
INDEX = PAPERS / "README.md"
START, END = "<!-- table:start -->", "<!-- table:end -->"
SECTIONS = ["## Recipe", "## Run", "## Result", "## Checks", "## Climb", "## Learned"]
HEADER = ["**Paper:**", "**Book:**", "**Claim:**", "**The change:**"]
KEYS = {
    "recipe",
    "paper",
    "base_model",
    "metric",
    "n_holdout",
    "k",
    "arms",
    "delta",
    "book",
    "checks",
    "gpu",
    "usd",
    "verified",
    "zeroproof",
}
ARM_KEYS = {"score", "ci", "steps"}
CHECK_KEYS = {
    "run_std",
    "decontaminated_dropped",
    "over_optimized",
    "length_before",
    "length_after",
    "hack_scan_top",
    "seed",
}
COLUMNS = (
    "| Recipe | Paper | Base | Metric | Baseline -> Recipe | Verified |\n|---|---|---|---|---|---|"
)


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def recipe_dirs() -> list[Path]:
    return [d for d in sorted(PAPERS.iterdir()) if d.is_dir() and not d.name.startswith("_")]


def load(d: Path) -> dict:
    return json.loads((d / "results.json").read_text(encoding="utf-8"))


def row(d: Path, r: dict) -> str:
    base, rec = r["arms"]["baseline"], r["arms"]["recipe"]
    delta = r["delta"]
    lo, hi = delta.get("ci", [0.0, 0.0])
    verified = "never run" if str(r["verified"]).startswith("1970") else r["verified"]
    paper_id = r["paper"].rstrip("/").rsplit("/", 1)[-1]
    return (
        f"| [{d.name}]({d.name}) | [{paper_id}]({r['paper']}) | {r['base_model']} "
        f"| {r['metric']} | {base['score']:.2f} -> {rec['score']:.2f} "
        f"({delta['recipe_vs_baseline']:+.2f} [{lo:+.2f}, {hi:+.2f}], {delta['verdict']}) "
        f"| {verified} |"
    )


def table(dirs: list[Path]) -> str:
    rows = [row(d, load(d)) for d in dirs] or ["| _none yet_ | | | | | |"]
    return "\n".join([START, COLUMNS, *rows, END])


def check_recipe(d: Path) -> dict:
    for name in ("README.md", "results.json", "recipe.py"):
        if not (d / name).exists():
            fail(f"{d.name}: missing {name}")
    text = (d / "README.md").read_text(encoding="utf-8")
    for s in HEADER + SECTIONS:
        if s not in text:
            fail(f"{d.name}: README missing '{s}'")
    if "<" in text.splitlines()[0]:
        fail(f"{d.name}: README title still has a placeholder")
    r = load(d)
    missing = KEYS - set(r)
    if missing:
        fail(f"{d.name}: results.json missing {sorted(missing)}")
    if r["recipe"] != d.name:
        fail(f"{d.name}: results.json recipe is '{r['recipe']}'")
    for arm in ("base", "baseline", "recipe"):
        if arm not in r["arms"]:
            fail(f"{d.name}: results.json arms missing '{arm}'")
        if ARM_KEYS - set(r["arms"][arm]):
            fail(f"{d.name}: arm '{arm}' missing {sorted(ARM_KEYS - set(r['arms'][arm]))}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r["verified"])):
        fail(f"{d.name}: verified must be YYYY-MM-DD")
    if r["delta"].get("verdict") not in ("moved", "flat"):
        fail(f"{d.name}: delta.verdict must be moved or flat")
    if not re.fullmatch(r"ch\. \d+.*", str(r["book"])):
        fail(f"{d.name}: book must name an rlhfbook.com chapter, like 'ch. 6'")
    if CHECK_KEYS - set(r["checks"]):
        fail(f"{d.name}: checks missing {sorted(CHECK_KEYS - set(r['checks']))}")
    # The science bar: "moved" needs an interval that excludes zero AND a delta
    # larger than twice the eval's own re-run noise (rlhf-book ch. 16, app. C),
    # and no over-optimization verdict (ch. 14). Otherwise it is "flat".
    if r["delta"]["verdict"] == "moved":
        lo, hi = r["delta"].get("ci", [0.0, 0.0])
        delta = float(r["delta"]["recipe_vs_baseline"])
        run_std = float(r["checks"]["run_std"])
        if lo <= 0.0 <= hi:
            fail(f"{d.name}: verdict moved but the interval [{lo}, {hi}] covers zero")
        if abs(delta) < 2.0 * run_std:
            fail(
                f"{d.name}: verdict moved but |delta| {abs(delta):.3f} < 2 x run_std {run_std:.3f}"
            )
        if r["checks"]["over_optimized"]:
            fail(f"{d.name}: verdict moved but the proxy-vs-target check says over-optimized")
    return r


def main(write: bool) -> None:
    dirs = recipe_dirs()
    for d in dirs:
        check_recipe(d)
        print(f"ok   {d.name}")
    index = INDEX.read_text(encoding="utf-8")
    if START not in index or END not in index:
        fail(f"{INDEX.name} needs the {START} and {END} markers")
    fresh = table(dirs)
    pre, rest = index.split(START, 1)
    _, post = rest.split(END, 1)
    new_index = pre + fresh + post
    if write and new_index != index:
        INDEX.write_text(new_index, encoding="utf-8")
        print("wrote the table")
    elif new_index != index:
        fail(f"table in {INDEX.name} is stale; run: python recipes/papers/check.py --write")
    print(f"ok   {len(dirs)} paper recipe(s)")


if __name__ == "__main__":
    main(write="--write" in sys.argv[1:])
