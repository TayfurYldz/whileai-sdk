"""Serve any Hugging Face model as an OpenAI-compatible endpoint on Modal (vLLM).

The account's hosted endpoint serves Qwen3-4B and Phi-4. To benchmark or train
another base (a Nemotron, a Llama, your own checkpoint) the SDK only needs an
OpenAI-compatible URL and a key, so:

    T2S_SERVE_MODEL=nvidia/Llama-3.1-Nemotron-Nano-8B-v1 T2S_SERVE_KEY=<any-secret> \\
        PYTHONUTF8=1 modal deploy serve_modal.py
    # -> https://<workspace>--t2s-serve-<slug>-serve.modal.run/v1

    VLLM_API_KEY=<the same secret> python rollout.py \\
        --agent "vllm:nvidia/Llama-3.1-Nemotron-Nano-8B-v1@https://<workspace>--t2s-serve-<slug>-serve.modal.run/v1" \\
        --system-prefix "detailed thinking on" --split holdout --k 4

`T2S_SERVE_ADAPTER=volume:<run_id>` serves a LoRA adapter from the training
volume on top of the base (`--enable-lora`), named `<base>-adapter`.
One L40S (24 GB models) or H100; scale-to-zero after 10 idle minutes.
"""

from __future__ import annotations

import os
import re
import subprocess

import modal

MODEL = os.environ.get("T2S_SERVE_MODEL") or "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
KEY = os.environ.get("T2S_SERVE_KEY") or "t2s-serve"
ADAPTER = os.environ.get("T2S_SERVE_ADAPTER") or ""  # "volume:<run_id>"
GPU = os.environ.get("T2S_SERVE_GPU") or "L40S"
SLUG = re.sub(r"[^a-z0-9]+", "-", MODEL.lower()).strip("-")[-40:]
MAX_LEN = int(os.environ.get("T2S_SERVE_MAX_LEN") or 16384)

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
    # Modal imports this module again inside the container, where the deploy
    # shell's variables are gone: everything the server needs rides in the secret.
    secrets=[
        modal.Secret.from_dict(
            {
                "VLLM_API_KEY": KEY,
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "T2S_SERVE_MODEL": MODEL,
                "T2S_SERVE_ADAPTER": ADAPTER,
                "T2S_SERVE_MAX_LEN": str(MAX_LEN),
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
