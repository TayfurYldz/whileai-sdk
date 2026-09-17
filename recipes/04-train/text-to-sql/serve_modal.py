"""Serve any Hugging Face model as an OpenAI-compatible endpoint on Modal (vLLM).

The account's hosted endpoint serves Qwen3-4B and Phi-4. To benchmark or train
another base (a Nemotron, a Llama, your own checkpoint) the SDK only needs an
OpenAI-compatible URL and a key, so:

    python serve_modal.py --model nvidia/Llama-3.1-Nemotron-Nano-8B-v1 --key <secret>   # writes serve_config.json
    PYTHONUTF8=1 modal deploy serve_modal.py
    python serve_modal.py --url                                                          # the endpoint

    VLLM_API_KEY=<secret> python rollout.py \\
        --agent "vllm:nvidia/Llama-3.1-Nemotron-Nano-8B-v1@https://...modal.run/v1" \\
        --system-prefix "detailed thinking on" --split holdout --k 4

`--adapter volume:<run_id>` serves a LoRA adapter from the training volume on
top of the base (`--enable-lora`), as model id `<base>-adapter`. Settings live in
serve_config.json (gitignored), not environment variables: Modal imports this
module again inside the container, and on some shells `modal deploy` does not
see the caller's variables either. The server runs with a tool-call parser
because the SDK sends drafted tool schemas with every request. One L40S
(models up to ~24 GB) or H100; scale-to-zero after 10 idle minutes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "serve_config.json"
_cfg: dict = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
MODEL = _cfg.get("model") or "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
KEY = _cfg.get("key") or "t2s-serve"
ADAPTER = _cfg.get("adapter") or ""  # "volume:<run_id>"
GPU = _cfg.get("gpu") or "L40S"
MAX_LEN = int(_cfg.get("max_len") or 16384)
# "llama3_json" for Llama-family models (Nemotron-Nano-8B-v1), "hermes" for Qwen.
TOOL_PARSER = _cfg.get("tool_parser") or ("hermes" if "qwen" in MODEL.lower() else "llama3_json")
SLUG = re.sub(r"[^a-z0-9]+", "-", MODEL.lower()).strip("-")[-40:]

app = modal.App(f"t2s-serve-{SLUG}")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.0", "huggingface_hub[hf_transfer]==0.34.4")
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "0",
        }
    )
)
hf_cache = modal.Volume.from_name("zeroproof-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("zeroproof-train-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * 60 * 60,
    scaledown_window=10 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/vol": runs_volume},
    # Everything the server needs rides in the secret: the container re-imports
    # this module without serve_config.json.
    secrets=[
        modal.Secret.from_dict(
            {
                "VLLM_API_KEY": KEY,
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "T2S_SERVE_MODEL": MODEL,
                "T2S_SERVE_ADAPTER": ADAPTER,
                "T2S_SERVE_MAX_LEN": str(MAX_LEN),
                "T2S_SERVE_TOOL_PARSER": TOOL_PARSER,
            }
        )
    ],
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    model = os.environ["T2S_SERVE_MODEL"]
    key = os.environ["VLLM_API_KEY"]
    adapter = os.environ.get("T2S_SERVE_ADAPTER") or ""
    max_len = os.environ.get("T2S_SERVE_MAX_LEN") or "16384"
    tool_parser = os.environ.get("T2S_SERVE_TOOL_PARSER") or "hermes"
    cmd = [
        "vllm",
        "serve",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--api-key",
        key,
        "--max-model-len",
        max_len,
        "--gpu-memory-utilization",
        "0.90",
        "--dtype",
        "bfloat16",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_parser,
    ]
    if adapter.startswith("volume:"):
        run_id = adapter.split(":", 1)[1]
        cmd += [
            "--enable-lora",
            "--lora-modules",
            f"{model}-adapter=/vol/{run_id}/adapter",
            "--max-lora-rank",
            "64",
        ]
    subprocess.Popen(cmd)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="write serve_config.json, then: modal deploy serve_modal.py"
    )
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--key", default=KEY)
    ap.add_argument(
        "--adapter", default=ADAPTER, help="volume:<run_id>: serve a LoRA from the training volume"
    )
    ap.add_argument("--gpu", default=GPU)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--tool-parser", default=TOOL_PARSER)
    ap.add_argument("--url", action="store_true", help="print the deployed endpoint URL and exit")
    a = ap.parse_args()
    if a.url:
        print(modal.Function.from_name(app.name, "serve").get_web_url())
        raise SystemExit(0)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "model": a.model,
                "key": a.key,
                "adapter": a.adapter,
                "gpu": a.gpu,
                "max_len": a.max_len,
                "tool_parser": a.tool_parser,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {CONFIG_PATH.name} for {a.model}; now: PYTHONUTF8=1 modal deploy serve_modal.py")
