"""<Recipe name>: <paper> in one file.

    python recipe.py                          # both arms, writes results.json
    python recipe.py --arm recipe --steps 200 # one arm, longer

Shape of every recipe:
  1. data():      tasks + a task-disjoint holdout (public data or a seeded env in this dir)
  2. train(arm):  "baseline" or "recipe"; the recipe arm is the baseline plus ONE change
  3. evaluate():  same holdout, k samples per task, before and after each arm
  4. results.json + the paired delta (zeroproof.simulations.delta_report) on the run page

Training runs on Modal (TRL + LoRA, see recipes/04-train/grpo/train_modal.py)
or through the hosted trainer (zps.train(..., method=, loss_type=, beta=, ...)).
Keep the default under 60 GPU minutes.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from importlib.metadata import version
from pathlib import Path

import zeroproof.simulations as zps

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-4B"
METRIC = "pass@1"


def data(seed: int) -> tuple[list[dict], list[dict]]:
    """Return (train_tasks, holdout_tasks). Holdout is split by task id, never by row."""
    raise NotImplementedError


def train(arm: str, tasks: list[dict], steps: int, seed: int) -> str:
    """Train one arm. Return an adapter path or a served model name."""
    raise NotImplementedError


def evaluate(model: str | None, holdout: list[dict], k: int) -> list[dict]:
    """k samples per holdout task from `model` (None = the untrained base), graded rows."""
    raise NotImplementedError


def summarize(rows: list[dict]) -> dict:
    p = zps.pass_at(rows)  # pass@1 with its task-bootstrap interval, pass@k, pass^k
    return {"score": p.pass_at_1, "ci": list(p.ci95 or (0.0, 0.0)), "pass_at_k": p.pass_at_k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_tasks, holdout = data(args.seed)
    base_rows = evaluate(None, holdout, args.k)
    results = {
        "recipe": HERE.name,
        "base_model": BASE_MODEL,
        "metric": METRIC,
        "n_holdout": len(holdout),
        "k": args.k,
        "arms": {"base": {**summarize(base_rows), "steps": 0, "gpu_minutes": 0}},
        "verified": date.today().isoformat(),
        "zeroproof": version("zeroproof"),
    }
    arm_rows: dict[str, list[dict]] = {}
    for arm in ["baseline", "recipe"] if args.arm == "both" else [args.arm]:
        model = train(arm, train_tasks, args.steps, args.seed)
        arm_rows[arm] = evaluate(model, holdout, args.k)
        results["arms"][arm] = {**summarize(arm_rows[arm]), "steps": args.steps}
    if len(arm_rows) == 2:
        d = zps.delta_report(arm_rows["baseline"], arm_rows["recipe"], target="pass_at_1")
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": "moved" if d["target_verdict"] == "moved" else "flat",
        }
        print(zps.format_delta_report(d))
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
