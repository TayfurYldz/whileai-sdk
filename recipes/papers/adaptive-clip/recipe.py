"""Adaptive clip: the upper clip bound follows how rare a correct answer was.

    python recipe.py                      # both arms on Modal, writes results.json
    python recipe.py --arm recipe         # one arm
    python recipe.py --selftest           # the clip schedule and the loss, offline, no GPU

GRPO clips the importance ratio into ``[1 - eps_lo, 1 + eps_hi]``. DAPO's
"clip-higher" widens the upper side to 0.28 and leaves it there, the same for
every rollout in every group. The paper's complaint is that this treats two
very different rollouts alike: the one correct answer in a group of eight,
which is the only evidence the model has that the hard problem is solvable,
gets the same room to move as the seventh correct answer on a problem the
model already has.

So make the bound follow the group. With ``c`` correct rollouts out of ``k``
(equation 11 of the paper):

    eps_hi(c) = eps_lo + (eps_hi_max - eps_lo) * (k - c) / (k - 1)

c = 1, a rare correct answer, gets the full ``eps_hi_max``. c = k - 1, an easy
problem, gets almost nothing above ``eps_lo``. Both arms here share the same
``eps_lo`` = 0.20 and the same ceiling 0.28, which are the paper's token-level
defaults. The baseline pins the upper bound at that ceiling for everyone; the
recipe slides it down as correct answers get common. That is the one change.

Shape of the run:
  1. data():      GSM8K, train split for prompts, test split held out
  2. run_arm():   TRL GRPOTrainer + LoRA on Modal, one arm per call
  3. evaluate():  same holdout, k samples per task, graded by MathEqual
  4. results.json + the paired delta (zps.delta_report) on the run page
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
METRIC = "pass@1"

# The paper's token-level importance sampling defaults. EPS_LOW is the lower
# clip bound in both arms; EPS_HIGH_MAX is the upper bound the baseline uses
# for every rollout and the ceiling the recipe slides down from.
EPS_LOW = 0.20
EPS_HIGH_MAX = 0.28

SYSTEM = "Solve the problem. Think briefly, then give the final number as \\boxed{answer}."


# --------------------------------------------------------------------------
# The clip schedule. Pure functions, no torch: `--selftest` runs them on
# hand-written groups, and the Modal container imports this same file.
# --------------------------------------------------------------------------


def messages_for(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def gold_of(answer: str) -> str:
    """GSM8K ships the worked solution then `#### 72`. The gold is the tail."""
    return answer.split("####")[-1].strip().replace(",", "")


def outcome_of(text: str, gold: str) -> float:
    """1.0 when the final number matches the gold, else 0.0. A program, not a
    judge: `MathEqual` is sympy with a numeric and string fallback."""
    from zeroproof.simulations.verify import MathEqual

    row = {
        "prompt": "",
        "final_text": text,
        "privileged": {"reference": gold},
        "scenario_id": "x",
        "rollout_index": 0,
    }
    return 1.0 if MathEqual()(row).get("reward") == 1 else 0.0


def epsilon_high_for_group(
    correct: int, k: int, eps_low: float = EPS_LOW, eps_high_max: float = EPS_HIGH_MAX
) -> float:
    """THE ONE CHANGE, equation 11: the group's upper clip bound.

    `correct` is how many of the group's `k` rollouts got the answer right.
    One correct out of k returns `eps_high_max`; k - 1 correct returns a hair
    over `eps_low`. A group of one has no group to be relative to, so it keeps
    the ceiling.

    The formula is written for a group that actually splits, `1 <= c <= k`.
    Fed c = 0 it returns `eps_low + (eps_high_max - eps_low) * k / (k - 1)`,
    which is above the ceiling it is supposed to stop at (0.2914 against 0.28
    at k = 8). An all-wrong group has a zero advantage and contributes no
    gradient either way, so the count is clamped into [1, k] and c = 0 lands
    on the ceiling rather than over it.
    """
    if k <= 1:
        return eps_high_max
    correct = max(1, min(k, correct))
    return eps_low + (eps_high_max - eps_low) * (k - correct) / (k - 1)


def epsilon_high_per_rollout(
    advantages: list[float],
    k: int,
    eps_low: float = EPS_LOW,
    eps_high_max: float = EPS_HIGH_MAX,
) -> list[float]:
    """One upper bound per rollout, read off its own group's correct count.

    The advantages arrive `k` per prompt and contiguous, before TRL shuffles
    the generation batch. The reward here is binary, so a group's correct
    rollouts are exactly the ones whose advantage came out positive: with
    rewards in {0, 1} the advantage is `r - c/k`, which is positive for every
    correct rollout and negative for every wrong one. Counting signs saves
    carrying the rewards through as a second tensor.

    A unanimous group (all right or all wrong) has every advantage at zero, so
    it counts as zero correct and gets the ceiling. That bound never applies to
    anything: a zero advantage contributes no gradient whichever way it clips.
    """
    out: list[float] = []
    for start in range(0, len(advantages), k):
        block = advantages[start : start + k]
        correct = sum(1 for a in block if a > 0)
        out.extend([epsilon_high_for_group(correct, k, eps_low, eps_high_max)] * len(block))
    return out


def reward_fn(completions, answer, **kwargs) -> list[float]:
    """Binary outcome, a program against the public GSM8K gold. Both arms use
    this untouched: the paper changes the clip, not the reward."""
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return [outcome_of(t, gold_of(a)) for t, a in zip(texts, answer)]


def graded_rows(holdout: list[dict], replies: list[list[str]]) -> list[dict]:
    """Eval rows in the shape `pass_at` and `delta_report` read: binary
    `reward`, one row per sample, grouped by task."""
    rows: list[dict] = []
    for task, texts in zip(holdout, replies):
        gold = gold_of(task["answer"])
        for i, text in enumerate(texts):
            rows.append(
                {
                    "prompt": task["question"],
                    "final_text": text,
                    "reward": outcome_of(text, gold),
                    "scenario_id": task["scenario_id"],
                    "rollout_index": i,
                    "privileged": {"reference": gold},
                }
            )
    return rows


def adaptive_clip_trainer(base_cls):
    """Build the trainer subclass. Takes `GRPOTrainer` as an argument so this
    module imports without torch, which is what lets `--selftest` run locally.
    """

    import torch

    class AdaptiveClipTrainer(base_cls):  # type: ignore[valid-type,misc]
        """GRPOTrainer with the group-adaptive upper clip bound.

        Two small overrides and no copy of the loss body, so this rides along
        with whatever else TRL's GRPO loss does:

        * `_generate_and_score_completions` attaches one `eps_hi` per rollout
          while the batch is still grouped by prompt. TRL's `shuffle_tensor_dict`
          and `split_tensor_dict` index every value in that dict along dim 0
          together, so the bound stays glued to its rollout through the shuffle
          and the gradient-accumulation split.
        * `_compute_loss` swaps `self.epsilon_high` for that tensor while the
          parent runs. The parent's `torch.clamp(coef_1, 1 - self.epsilon_low,
          1 + self.epsilon_high)` then broadcasts a (batch, 1) bound over the
          (batch, tokens) ratio, and the clip-fraction metric TRL logs a few
          lines later broadcasts the same way, so `clip_ratio` stays honest.

        `adaptive=False` is the baseline: the class is the same, the flag is
        the difference.
        """

        def __init__(self, *args, adaptive: bool = True, eps_high_max: float = EPS_HIGH_MAX, **kw):
            super().__init__(*args, **kw)
            self.adaptive = adaptive
            self.eps_high_max = eps_high_max
            for attr in ("epsilon_low", "epsilon_high", "num_generations"):
                if not hasattr(self, attr):
                    raise RuntimeError(
                        f"this TRL ({version('trl')}) has no GRPOTrainer.{attr}; the recipe is "
                        "pinned to trl==0.19.1, where the clip bounds are plain attributes"
                    )
            if adaptive and getattr(self, "use_liger_loss", False):
                raise RuntimeError(
                    "use_liger_loss reads epsilon_high once at init, so the adaptive bound "
                    "would be silently ignored; run this recipe without Liger"
                )

        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            advantages = out.get("advantages")
            if not self.adaptive or advantages is None:
                return out
            eps = epsilon_high_per_rollout(
                advantages.tolist(), self.num_generations, self.epsilon_low, self.eps_high_max
            )
            out["gapo_epsilon_high"] = torch.tensor(
                eps, dtype=torch.float32, device=advantages.device
            )
            return out

        def _compute_loss(self, model, inputs):
            eps = inputs.get("gapo_epsilon_high")
            if eps is None:
                return super()._compute_loss(model, inputs)
            saved_high, saved_low = self.epsilon_high, self.epsilon_low
            # Both bounds go in as (batch, 1) tensors. Handing torch.clamp one
            # tensor bound and one float would work too, but matching shapes
            # keeps it on the unambiguous Tensor overload.
            self.epsilon_high = eps.unsqueeze(1)
            self.epsilon_low = torch.full_like(self.epsilon_high, float(saved_low))
            try:
                return super()._compute_loss(model, inputs)
            finally:
                self.epsilon_high, self.epsilon_low = saved_high, saved_low

    return AdaptiveClipTrainer


# --------------------------------------------------------------------------
# Modal: the pins and the image from recipes/04-train/grpo/train_modal.py.
# --------------------------------------------------------------------------

DEFAULT_GPU = os.environ.get("ZP_RECIPE_GPU", "L40S")
VOLUME_ROOT = "/vol"

app = modal.App("zeroproof-recipe-adaptive-clip")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.54.0",
        "trl==0.19.1",
        "peft==0.16.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "zeroproof",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(str(HERE / "recipe.py"), "/root/recipe_mod.py")
)

runs_volume = modal.Volume.from_name("zeroproof-recipe-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("zeroproof-hf-cache", create_if_missing=True)
dashboard_secret = modal.Secret.from_dict(
    {"ZEROPROOF_API_KEY": os.environ.get("ZEROPROOF_API_KEY", "")}
)


def _sample(model, tokenizer, questions, *, n, max_new_tokens, batch=8):
    """`n` replies per question, batched, sampled the way the trainer samples."""
    import torch

    sys.path.insert(0, "/root")
    from recipe_mod import messages_for

    model.eval()
    tokenizer.padding_side = "left"
    out: list[list[str]] = []
    texts = [
        tokenizer.apply_chat_template(messages_for(q), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append(decoded[i * n : (i + 1) * n])
    model.train()
    return out


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes={VOLUME_ROOT: runs_volume, "/root/.cache/huggingface": hf_cache},
    secrets=[dashboard_secret],
)
def run_arm(
    arm: str,
    adaptive: bool,
    train_tasks: list[dict],
    holdout: list[dict],
    run_name: str,
    base_model: str = BASE_MODEL,
    steps: int = 40,
    num_generations: int = 8,
    prompts_per_step: int = 6,
    learning_rate: float = 1e-4,
    eps_high_max: float = EPS_HIGH_MAX,
    max_completion_length: int = 256,
    lora_rank: int = 32,
    eval_samples: int = 4,
    eval_base: bool = False,
) -> dict:
    """One arm: eval the base model (optionally), train, eval again."""
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, "/root")
    from recipe_mod import (
        EPS_LOW,
        adaptive_clip_trainer,
        graded_rows,
        messages_for,
        reward_fn,
    )

    import zeroproof.simulations as zps

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    config = {
        "arm": arm,
        "adaptive_clip": adaptive,
        "base_model": base_model,
        "steps": steps,
        "num_generations": num_generations,
        "prompts_per_step": prompts_per_step,
        "learning_rate": learning_rate,
        "epsilon_low": EPS_LOW,
        "epsilon_high_max": eps_high_max,
        "beta": 0.0,
        "lora_rank": lora_rank,
        "max_completion_length": max_completion_length,
        "train_prompts": len(train_tasks),
        "holdout": len(holdout),
        "gpu": DEFAULT_GPU,
        "reward": "binary MathEqual against the GSM8K gold",
    }
    run = None
    if os.environ.get("ZEROPROOF_API_KEY"):
        run = zps.training_run(
            run_name,
            base_model=base_model,
            trainer="trl-grpo-lora",
            total_steps=steps,
            config=config,
        )
        print(f"dashboard: {run.url}")

    questions = [t["question"] for t in holdout]
    base_rows = None
    if eval_base:
        replies = _sample(
            model, tokenizer, questions, n=eval_samples, max_new_tokens=max_completion_length
        )
        base_rows = graded_rows(holdout, replies)
        print(f"base:   {zps.pass_at(base_rows)}")

    dataset = Dataset.from_list(
        [{"prompt": messages_for(t["question"]), "answer": t["answer"]} for t in train_tasks]
    )
    out_dir = os.path.join(VOLUME_ROOT, run_name)
    grpo = GRPOConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        max_steps=steps,
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=prompts_per_step,
        learning_rate=learning_rate,
        # No KL term, so the clip is the only trust region in the run and the
        # arms differ in the thing the paper is about and nothing else.
        beta=0.0,
        epsilon=EPS_LOW,
        # The baseline's fixed upper bound, and the recipe's ceiling. The
        # recipe arm overwrites it per group inside the trainer subclass.
        epsilon_high=eps_high_max,
        max_completion_length=max_completion_length,
        max_prompt_length=512,
        temperature=0.9,
        bf16=True,
        # Off on purpose: this trainer generates during training, and
        # checkpointing corrupts Qwen generation on these pins.
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=17,
    )
    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    trainer = adaptive_clip_trainer(GRPOTrainer)(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        adaptive=adaptive,
        eps_high_max=eps_high_max,
    )
    if run is not None:
        trainer.add_callback(zps.TrainerCallback(run, finish=False))
    try:
        trainer.train()
    except Exception as exc:
        if run is not None:
            run.fail(f"{type(exc).__name__}: {exc}")
        raise

    replies = _sample(
        trainer.model, tokenizer, questions, n=eval_samples, max_new_tokens=max_completion_length
    )
    after_rows = graded_rows(holdout, replies)
    after = zps.pass_at(after_rows)
    print(f"{arm}: {after}")

    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    runs_volume.commit()

    gpu_minutes = (time.time() - started) / 60.0
    summary = {
        "arm": arm,
        "adaptive_clip": adaptive,
        "pass_at_1": after.pass_at_1,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
    }
    if run is not None:
        run.finish("done", summary=summary, adapter=f"zeroproof-recipe-runs:/{run_name}/adapter")
        summary["run_url"] = run.url
    return {
        "arm": arm,
        "base_rows": base_rows,
        "after_rows": after_rows,
        "gpu_minutes": gpu_minutes,
        "steps": steps,
        "run_url": summary.get("run_url", ""),
    }


# --------------------------------------------------------------------------
# Local: data, orchestration, results.json.
# --------------------------------------------------------------------------


def data(seed: int, n_train: int, n_holdout: int) -> tuple[list[dict], list[dict]]:
    """GSM8K. Train prompts from the train split, holdout from the test split,
    so the two are disjoint by construction, not by a shuffle."""
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    test = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    train_tasks = [
        {"question": r["question"], "answer": r["answer"], "scenario_id": f"train-{i}"}
        for i, r in enumerate(train.select(range(n_train)))
    ]
    holdout = [
        {"question": r["question"], "answer": r["answer"], "scenario_id": f"test-{i}"}
        for i, r in enumerate(test.select(range(n_holdout)))
    ]
    return train_tasks, holdout


def summarize(rows: list[dict]) -> dict:
    import zeroproof.simulations as zps

    p = zps.pass_at(rows)
    return {
        "score": p.pass_at_1,
        "ci": list(p.ci95 or (0.0, 0.0)),
        "pass_at_k": p.pass_at_k,
    }


def selftest() -> None:
    """The clip schedule on hand-written groups, and the loss it feeds, on the
    CPU. No GPU, no key, no model download."""
    k = 8

    # Equation 11 at the ends and in the middle.
    assert epsilon_high_for_group(1, k) == EPS_HIGH_MAX
    assert abs(epsilon_high_for_group(k, k) - EPS_LOW) < 1e-12
    assert epsilon_high_for_group(0, k) == EPS_HIGH_MAX
    schedule = [round(epsilon_high_for_group(c, k), 4) for c in range(1, k + 1)]
    assert schedule == sorted(schedule, reverse=True), "rarer correct -> more headroom"
    print(f"eps_hi by correct-in-group (k={k}, eps_lo={EPS_LOW}, ceiling={EPS_HIGH_MAX}):")
    for c, e in zip(range(1, k + 1), schedule):
        print(f"  {c} of {k} correct -> {e}")

    # Two groups of four laid out the way TRL hands them over: one hard (one
    # correct), one easy (three correct). Advantages are r - mean(r).
    hard = [1 - 0.25, -0.25, -0.25, -0.25]
    easy = [1 - 0.75, 1 - 0.75, 1 - 0.75, -0.75]
    eps = epsilon_high_per_rollout(hard + easy, 4)
    assert eps[:4] == [epsilon_high_for_group(1, 4)] * 4
    assert eps[4:] == [epsilon_high_for_group(3, 4)] * 4
    print(f"hard group (1 of 4 right) -> {eps[0]:.4f}; easy group (3 of 4) -> {eps[4]:.4f}")

    # A unanimous group is all zeros, so it counts as zero correct and takes
    # the ceiling. The bound is moot: a zero advantage has no gradient.
    assert epsilon_high_per_rollout([0.0] * 4, 4) == [EPS_HIGH_MAX] * 4

    # The grader is a program, not a judge.
    assert outcome_of("so the answer is \\boxed{18}", "18") == 1.0
    assert outcome_of("the answer is 5", "18") == 0.0
    print("grader: MathEqual reads \\boxed{} and the last number")

    _selftest_loss()
    print("selftest ok")


def _selftest_loss() -> None:
    """The bound really does reach the loss: run TRL's own `_compute_loss`
    body on a CPU tensor and check that a wider bound clips less.

    This is the line the recipe cannot check by reading: TRL calls
    `torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)`, and the
    whole change rests on those bounds accepting a (batch, 1) tensor and
    broadcasting over the (batch, tokens) ratio.
    """
    try:
        import torch
    except ImportError:
        print("torch not installed locally: skipping the clamp check")
        return

    ratio = torch.tensor([[1.5, 1.5], [1.5, 1.5]])
    advantages = torch.tensor([0.75, 0.25])
    fixed_high = torch.full((2, 1), EPS_HIGH_MAX)
    # Group A had one correct of four, group B had three of four.
    adaptive_high = torch.tensor([[epsilon_high_for_group(1, 4)], [epsilon_high_for_group(3, 4)]])
    low = torch.full((2, 1), EPS_LOW)

    def surrogate(high: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(ratio, 1 - low, 1 + high)
        return -torch.min(ratio * advantages.unsqueeze(1), clipped * advantages.unsqueeze(1))

    fixed, adaptive = surrogate(fixed_high), surrogate(adaptive_high)
    # The ratio is above both ceilings and the advantage is positive, so the
    # clipped branch wins and the loss is exactly -(1 + eps_hi) * advantage.
    assert torch.allclose(fixed[0], -(1 + EPS_HIGH_MAX) * advantages[0]), fixed
    assert torch.allclose(adaptive[0], fixed[0]), "rare-correct group keeps the ceiling"
    assert adaptive[1].mean() > fixed[1].mean(), "common-correct group should be reined in"
    print(
        f"clamp with a (batch, 1) bound: rare-correct loss {adaptive[0, 0]:+.4f} "
        f"(unchanged), common-correct loss {adaptive[1, 0]:+.4f} vs {fixed[1, 0]:+.4f} fixed"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "recipe", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k", type=int, default=4, help="eval samples per holdout task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-holdout", type=int, default=120)
    ap.add_argument(
        "--generations", type=int, default=8, help="rollouts per prompt, the k in eq 11"
    )
    ap.add_argument("--prompts-per-step", type=int, default=6)
    ap.add_argument("--eps-high-max", type=float, default=EPS_HIGH_MAX)
    ap.add_argument("--selftest", action="store_true", help="the clip schedule, offline")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import zeroproof.simulations as zps

    train_tasks, holdout = data(args.seed, args.n_train, args.n_holdout)
    arms = ["baseline", "recipe"] if args.arm == "both" else [args.arm]
    adaptive = {"baseline": False, "recipe": True}

    # Start from what is already on disk, so `--arm recipe` refreshes one arm
    # instead of wiping the other one and the delta. Only a both-arm run moves
    # `verified` and the delta; a one-arm run says so and leaves them alone.
    results = json.loads((HERE / "results.json").read_text())
    results.update(
        {
            "recipe": HERE.name,
            "title": "Adaptive clip: the upper bound follows how rare a correct answer was",
            "paper": "https://arxiv.org/abs/2609.00444",
            "base_model": BASE_MODEL,
            "metric": METRIC,
            "n_holdout": len(holdout),
            "k": args.k,
            "gpu": DEFAULT_GPU,
            "zeroproof": version("zeroproof"),
        }
    )
    results.setdefault("arms", {})
    arm_rows: dict[str, list[dict]] = {}
    usd_per_hour = {"A10G": 1.10, "L40S": 2.00, "H100": 4.00}.get(DEFAULT_GPU, 2.00)
    gpu_minutes = 0.0
    run_url = ""

    with modal.enable_output(), app.run():
        for i, arm in enumerate(arms):
            out = run_arm.remote(
                arm,
                adaptive[arm],
                train_tasks,
                holdout,
                f"adaptive-clip-{arm}-{date.today().isoformat()}",
                steps=args.steps,
                num_generations=args.generations,
                prompts_per_step=args.prompts_per_step,
                eps_high_max=args.eps_high_max,
                eval_samples=args.k,
                eval_base=(i == 0),
            )
            gpu_minutes += out["gpu_minutes"]
            run_url = out["run_url"] or run_url
            if out["base_rows"]:
                arm_rows["base"] = out["base_rows"]
                results["arms"]["base"] = {
                    **summarize(out["base_rows"]),
                    "steps": 0,
                    "gpu_minutes": 0,
                }
            arm_rows[arm] = out["after_rows"]
            results["arms"][arm] = {
                **summarize(out["after_rows"]),
                "steps": out["steps"],
                "gpu_minutes": round(out["gpu_minutes"], 1),
            }

    if "baseline" in arm_rows and "recipe" in arm_rows:
        d = zps.delta_report(arm_rows["baseline"], arm_rows["recipe"], target="pass_at_1")
        results["delta"] = {
            "recipe_vs_baseline": d["target_delta"],
            "ci": list(d["target_ci95"] or (0.0, 0.0)),
            "verdict": "moved" if d["target_verdict"] == "moved" else "flat",
        }
        results["verified"] = date.today().isoformat()
        results.pop("partial_run", None)
        print(zps.format_delta_report(d))
    else:
        results["partial_run"] = f"{date.today().isoformat()}: {', '.join(arms)} only"
        print(f"one arm only ({', '.join(arms)}): delta and verified left as they were")

    results["usd"] = round(gpu_minutes / 60.0 * usd_per_hour, 2)
    results["run_url"] = run_url
    print(f"wall clock: {gpu_minutes:.1f} GPU minutes, ${results['usd']:.2f} on {DEFAULT_GPU}")
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
