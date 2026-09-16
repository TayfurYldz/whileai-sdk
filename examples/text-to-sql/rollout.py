"""Run a policy over the tasks, k times each, and save SDK-shaped rows.

Usage:
  python rollout.py --model qwen3-4b   [--k 4] [--split all|holdout|train] [--concurrency 8]
  python rollout.py --model sonnet-5
  python rollout.py --model haiku-4.5

Rows land in raw/<model>.jsonl; the run is resumable by (task, rollout_index).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2s import RAW, load_tasks, make_row, read_jsonl, split_of, system_prompt

MODELS = {
    "qwen3-4b": {
        "kind": "openai",
        "model": "Qwen/Qwen3-4B",
        "base_url": "https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1",
        "key_env": "ZEROPROOF_API_KEY",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "qwen3-4b-think": {
        "kind": "openai",
        "model": "Qwen/Qwen3-4B",
        "base_url": "https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1",
        "key_env": "ZEROPROOF_API_KEY",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "max_tokens": 4000,
    },
    # Sonnet 5 rejects `temperature` (400: deprecated for this model); it samples at its default.
    "sonnet-5": {
        "kind": "bedrock",
        "model": "global.anthropic.claude-sonnet-5",
        "api_model": "claude-sonnet-5",
        "temperature_ok": False,
    },
    "haiku-4.5": {
        "kind": "bedrock",
        "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "api_model": "claude-haiku-4-5-20251001",
    },
}


def user_env(name: str) -> str:
    v = os.environ.get(name)
    if v:
        return v
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-c",
            f'[Environment]::GetEnvironmentVariable("{name}","User")',
        ],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def make_caller(spec: dict, temperature: float, max_tokens: int):
    sys_p = system_prompt()
    if spec["kind"] == "openai":
        from openai import OpenAI

        client = OpenAI(
            base_url=spec["base_url"],
            api_key=user_env(spec["key_env"]),
            max_retries=3,
            timeout=600.0,
        )

        def call(question: str) -> tuple[str, str | None, dict | None]:
            resp = client.chat.completions.create(
                model=spec["model"],
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": question},
                ],
                temperature=temperature,
                max_tokens=spec.get("max_tokens", max_tokens),
                extra_body=spec.get("extra_body"),
            )
            ch = resp.choices[0]
            text = ch.message.content or ""
            usage = resp.usage.model_dump() if resp.usage else None
            return text, ch.finish_reason, usage

        return call

    import anthropic

    # ANTHROPIC_API_KEY -> the Claude API (model ids without the Bedrock prefix); otherwise Bedrock.
    if os.environ.get("ANTHROPIC_API_KEY"):
        client = anthropic.Anthropic(max_retries=4, timeout=120.0)
        spec = dict(spec, model=spec.get("api_model", spec["model"]))
    else:
        client = anthropic.AnthropicBedrock(
            aws_region=os.environ.get("AWS_REGION") or "us-west-2", max_retries=4, timeout=120.0
        )

    def call_b(question: str) -> tuple[str, str | None, dict | None]:
        extra = {"temperature": temperature} if spec.get("temperature_ok", True) else None
        resp = client.messages.create(
            model=spec["model"],
            max_tokens=max_tokens,
            system=[{"type": "text", "text": sys_p, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": question}],
            extra_body=extra,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        stop = {"end_turn": "stop", "max_tokens": "length"}.get(
            resp.stop_reason or "", resp.stop_reason
        )
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0),
        }
        return text, stop, usage

    return call_b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument(
        "--hosted",
        default="",
        help="name of a model served from the account; rows land in raw/hosted-<name>.jsonl",
    )
    ap.add_argument(
        "--think", action="store_true", help="with --hosted: thinking on, max_tokens 4000"
    )
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--split", default="all", choices=["all", "holdout", "train"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke)")
    args = ap.parse_args()

    spec = MODELS[args.model]
    if args.hosted:
        # a model served from the account (zps.serve): same endpoint, model = the hosted name
        spec = dict(MODELS["qwen3-4b-think" if args.think else "qwen3-4b"], model=args.hosted)
        args.model = f"hosted-{args.hosted}" + ("-think" if args.think else "")
    tasks = load_tasks()
    if args.split != "all":
        tasks = [t for t in tasks if split_of(t["id"]) == args.split]
    if args.limit:
        tasks = tasks[: args.limit]
    out_path = RAW / f"{args.model}.jsonl"
    have = read_jsonl(out_path)
    done = {(r["scenario_id"], r["rollout_index"]) for r in have}
    jobs = [(t, i) for t in tasks for i in range(args.k) if (t["id"], i) not in done]
    print(
        f"{args.model}: {len(tasks)} tasks x k={args.k}; {len(have)} rows on disk; {len(jobs)} to run",
        flush=True,
    )
    if not jobs:
        return 0

    call = make_caller(spec, args.temperature, args.max_tokens)
    lock = threading.Lock()
    RAW.mkdir(exist_ok=True)
    f = out_path.open("a", encoding="utf-8")
    t0 = time.time()
    errors = 0

    def run(job):
        task, i = job
        t1 = time.time()
        text, finish, usage = call(task["question"])
        return make_row(
            task,
            i,
            args.model,
            text,
            finish_reason=finish,
            usage=usage,
            latency_s=round(time.time() - t1, 2),
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(run, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as exc:
                errors += 1
                print(f"  error {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                continue
            with lock:
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}  {time.time() - t0:.0f}s  errors={errors}", flush=True)
    f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
