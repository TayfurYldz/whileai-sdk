"""The character example runs offline end to end, its rows validate, and
the judge check, pairs and before/after measurement behave."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import zeroproof.simulations as zps

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "character"


def _load(name: str):
    path = EXAMPLE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"character_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(f"character_example_{name}", module)
    spec.loader.exec_module(module)
    return module


RUN = _load("run")
PARSE = _load("from_model_spec")


def test_constitution_is_labeled_and_traceable():
    doc = json.loads((EXAMPLE / "constitution.json").read_text(encoding="utf-8"))
    assert doc["source"]["repo"] == "https://github.com/openai/model_spec"
    assert doc["source"]["license"] == "CC0-1.0"
    assert len(doc["traits"]) == 8
    for trait in doc["traits"]:
        assert trait["principle"]
        assert trait["examples"], trait["id"]
        for ex in trait["examples"]:
            assert ex["good"] and ex["bad"] and ex["prompt"]
        assert "[?]" not in trait["principle"]


def test_parser_reads_comparisons_and_prefix():
    text = """## Be warm {#be_warm authority=guideline}

The assistant is warm[^abcd] (see [?](#other)).

**Example**: x

~~~xml
<user>
hello
</user>
<assistant>
hi there
</assistant>
<user>
how are you
</user>
<comparison>
<assistant> <!-- GOOD: kind -->
good reply &amp; more
</assistant>
<assistant> <!-- BAD: cold -->
bad reply
</assistant>
</comparison>
~~~

## Next {#next authority=user}
"""
    built = PARSE.build(text, ["be_warm", "missing_one"])
    assert built["missing"] == ["missing_one"]
    (trait,) = built["traits"]
    assert trait["principle"] == "The assistant is warm."
    (ex,) = trait["examples"]
    assert ex["prompt"] == "how are you"
    assert ex["good"] == ["good reply & more"]
    assert ex["bad"] == ["bad reply"]
    assert [m["role"] for m in ex["prefix"]] == ["user", "assistant"]


def test_offline_run_is_deterministic_and_valid(tmp_path):
    a = RUN.run(out=tmp_path / "a", k=4, seed=1)
    b = RUN.run(out=tmp_path / "b", k=4, seed=1)
    assert a["pass_at"] == b["pass_at"]
    assert a["counts"] == b["counts"]
    rows = [json.loads(line) for line in (tmp_path / "a" / "rows.jsonl").open(encoding="utf-8")]
    assert rows and all(not zps.validate(r) for r in rows)
    trait_rows = [r for r in rows if r["trait"]]
    assert all(r["spec_id"].startswith("model_spec#") for r in trait_rows)
    assert all(r["privileged"]["principle"] for r in trait_rows)
    controls = [r for r in rows if r["split"] == "control"]
    assert controls and all("trait" not in r["markers"] for r in controls)
    assert all(r["markers"]["on_task"] == 1.0 for r in controls)
    # The reference judge is a lookup against the spec's labels, so it
    # must agree with them completely; a live judge reports a real number.
    assert a["judge_vs_spec"]["agreement"] == 1.0
    assert a["counts"]["spec_rows"] == a["judge_vs_spec"]["n"]


def test_offline_run_has_contrast_and_exports(tmp_path):
    rep = RUN.run(out=tmp_path, k=4, seed=0)
    assert rep["group_signal"]["n_mixed"] >= 1
    assert 0.0 < rep["pass_at"]["pass_at_1"] < 1.0
    assert rep["exports"]["pairs"]["pairs"] >= 1
    pairs = [json.loads(line) for line in (tmp_path / "pairs.jsonl").open(encoding="utf-8")]
    assert all(not zps.validate(p, "preference") for p in pairs)
    assert all(p["chosen"][0] == {"role": "system", "content": RUN.DEPLOY_PROMPT} for p in pairs)
    assert all(p["margin"] == 1.0 for p in pairs)
    sft = [json.loads(line) for line in (tmp_path / "sft.jsonl").open(encoding="utf-8")]
    assert sft and all(not zps.validate(s, "training") for s in sft)
    assert all(s["loss_mask"][-1] == 1 and s["loss_mask"][0] == 0 for s in sft)
    assert rep["decontaminate"]["n_contaminated"] == 0


def test_measure_demo_moves_trait_without_regressions():
    measure = _load("measure")
    before, after = measure.demo(seed=0)
    rep = measure.measure(before, after, seed=0)
    assert rep["ok"], rep.get("warnings")
    assert rep["target_verdict"] == "moved"
    assert rep["metrics"]["marker:trait"]["delta"] > 0
    assert not rep["regressions"]


@pytest.mark.parametrize("after", [False, True])
def test_scripted_student_filler_and_propensity(after):
    constitution = RUN.load_constitution()
    tasks = RUN.build_tasks(constitution)
    replies = [RUN.scripted_student(t, i, seed=0, after=after) for t in tasks for i in range(4)]
    filler = sum(RUN.has_filler(r) for r in replies) / len(replies)
    assert (filler < 0.15) if after else (0.2 < filler < 0.5)
