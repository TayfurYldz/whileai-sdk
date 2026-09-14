"""Training runs: the loss curve and the progress bar on the platform.

The SDK does not train. Your trainer does, wherever it runs, and this
module is how it reports: a run is created, points are logged as it goes,
and it is finished with a status. The platform draws the curve and the
progress bar at zeroproofai.com/platform/training.

Three ways in, one record:

* Engineer, one line. ``trainer.add_callback(zps.TrainerCallback(run))``
  on a Transformers or TRL trainer logs every ``on_log`` (loss, learning
  rate, eval loss, epoch, grad norm), sets the step count from the
  trainer, and finishes the run when training ends or crashes.
* Data scientist with a loop. ``run = zps.training_run("sft-v3",
  dataset="ds_...")``, then ``run.log(step, loss=...)`` wherever the loop
  has a number, ``run.finish()`` at the end. Points are buffered and sent
  in batches; logging never raises into the training loop.
* Researcher with a stack. Plain HTTP: ``POST /runs``, ``POST
  /runs/{id}/log`` with ``{"points": [{"step": 10, "loss": 1.2}]}``,
  ``POST /runs/{id}/finish``. The README lists the bodies.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .ingest.platform import _call

log = logging.getLogger("zeroproof.simulations")

SITE_URL = "https://www.zeroproofai.com"


def _json_safe(value: Any) -> Any:
    """Tuples to lists, NaN/inf to None, so a report survives JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


FLUSH_EVERY = 25
FLUSH_SECONDS = 15.0
MAX_BATCH = 500


class TrainingRun:
    """One fine-tune, as the platform sees it. Create with ``training_run``.

    ``log`` buffers; ``flush`` sends. A send that fails is retried on the
    next flush and counted in ``errors``; the training loop is never
    interrupted by the dashboard. ``finish`` flushes first.
    """

    def __init__(
        self,
        run_id: str,
        *,
        name: str,
        api_key: str | None = None,
        total_steps: int | None = None,
        flush_every: int = FLUSH_EVERY,
        flush_seconds: float = FLUSH_SECONDS,
        transport: Callable[..., Any] | None = None,
    ):
        self.run_id = run_id
        self.name = name
        self.total_steps = total_steps
        self.status = "running"
        self.step = 0
        self.errors = 0
        self._api_key = api_key
        self._flush_every = max(1, int(flush_every))
        self._flush_seconds = float(flush_seconds)
        self._call = transport or _call
        self._buffer: list[dict[str, Any]] = []
        self._pending_total: int | None = None
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        self._warned = False
        self._delta: dict[str, Any] | None = None
        self._summary: dict[str, Any] = {}

    @property
    def url(self) -> str:
        return f"{SITE_URL}/platform/training/{self.run_id}"

    # ------------------------------------------------------------ logging

    def log(self, step: int, **metrics: float) -> None:
        """Record one point. Any finite numeric keyword is a metric
        (``loss``, ``eval_loss``, ``lr``, ``epoch``, ``grad_norm``, ...)."""
        point: dict[str, Any] = {"step": int(step), "ts": time.time()}
        for key, value in metrics.items():
            if isinstance(value, bool) or value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number or number in (float("inf"), float("-inf")):
                continue
            point[str(key)] = number
        with self._lock:
            self._buffer.append(point)
            self.step = max(self.step, int(step))
            due = (
                len(self._buffer) >= self._flush_every
                or time.monotonic() - self._last_flush >= self._flush_seconds
            )
        if due:
            self.flush()

    def progress(self, step: int, total_steps: int | None = None) -> None:
        """Advance the bar without a metric. ``total_steps`` (re)sets the
        denominator; a trainer that learns its length late can call this."""
        if total_steps:
            with self._lock:
                self.total_steps = int(total_steps)
                self._pending_total = int(total_steps)
        self.log(step)

    def flush(self) -> bool:
        """Send buffered points. Returns True when nothing is left unsent."""
        with self._lock:
            batch = self._buffer[:MAX_BATCH]
            total = self._pending_total
        if not batch and total is None:
            return True
        body: dict[str, Any] = {"points": batch or [{"step": self.step, "ts": time.time()}]}
        if total is not None:
            body["total_steps"] = total
        try:
            self._call("POST", f"/runs/{self.run_id}/log", self._api_key, body)
        except Exception as exc:  # the dashboard must never stop the trainer
            self.errors += 1
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"training run {self.run_id}: could not send points ({exc}); "
                    "will retry on the next flush",
                    stacklevel=2,
                )
            log.debug("training run %s flush failed: %s", self.run_id, exc)
            return False
        with self._lock:
            del self._buffer[: len(batch)]
            if total is not None and self._pending_total == total:
                self._pending_total = None
            self._last_flush = time.monotonic()
            remaining = bool(self._buffer)
        if remaining:
            return self.flush()
        return True

    # ------------------------------------------------------------ lifecycle

    def finish(
        self,
        status: str = "done",
        *,
        summary: Mapping[str, Any] | None = None,
        adapter: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Flush, then mark the run ``done``, ``failed``, or ``stopped``."""
        self.flush()
        body: dict[str, Any] = {"status": status}
        merged = dict(summary or {})
        if self._delta is not None:
            merged["delta"] = self._delta
        if merged:
            body["summary"] = _json_safe(merged)
            self._summary = dict(merged)
        if adapter:
            body["adapter"] = str(adapter)
        if error:
            body["error"] = str(error)[:2000]
        try:
            out = self._call("POST", f"/runs/{self.run_id}/finish", self._api_key, body)
        except Exception as exc:
            self.errors += 1
            warnings.warn(f"training run {self.run_id}: could not finish ({exc})", stacklevel=2)
            out = {"runId": self.run_id, "status": status, "unsent": True}
        self.status = status
        return out if isinstance(out, dict) else {"runId": self.run_id, "status": status}

    def fail(self, error: str) -> dict[str, Any]:
        return self.finish("failed", error=error)

    def delta(
        self,
        before: Sequence[dict],
        after: Sequence[dict],
        *,
        target: str | None = "pass_at_1",
        must_not_regress: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Did the training move the behavior? ``delta_report`` over the
        rollouts before and after, kept on the run and sent with
        ``finish`` under ``summary["delta"]`` (sent right away when the run
        is already finished). The run page draws it."""
        from .score.delta import delta_report

        report = delta_report(before, after, target=target, must_not_regress=list(must_not_regress))
        self._delta = _json_safe(report)
        if self.status != "running":
            self._send_delta()
        return report

    def _send_delta(self) -> None:
        try:
            self._call(
                "POST",
                f"/runs/{self.run_id}/finish",
                self._api_key,
                {"status": self.status, "summary": {**self._summary, "delta": self._delta}},
            )
        except Exception as exc:
            self.errors += 1
            warnings.warn(f"training run {self.run_id}: could not send delta ({exc})", stacklevel=2)

    def __enter__(self) -> TrainingRun:
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if self.status != "running":
            return
        if exc is not None:
            self.finish("failed", error=f"{exc_type.__name__}: {exc}")
        else:
            self.finish("done")

    def __repr__(self) -> str:
        return (
            f"TrainingRun({self.run_id!r}, {self.name!r}, status={self.status!r}, step={self.step})"
        )


def training_run(
    name: str,
    *,
    dataset: str | None = None,
    after_dataset: str | None = None,
    base_model: str | None = None,
    trainer: str | None = None,
    total_steps: int | None = None,
    config: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    flush_every: int = FLUSH_EVERY,
    flush_seconds: float = FLUSH_SECONDS,
    transport: Callable[..., Any] | None = None,
) -> TrainingRun:
    """Create a run on the platform and return the handle to log into.

    ``dataset`` is the ``ds_...`` id trained on; ``after_dataset`` the set
    of post-training rollouts, when you have one, so the run page can show
    the before/after. ``config`` is anything JSON-shaped you want on the
    run page (hyperparameters, the command). ``api_key`` defaults to the
    usual credential chain.
    """
    body: dict[str, Any] = {"name": name}
    if dataset:
        body["dataset_id"] = dataset
    if after_dataset:
        body["after_dataset_id"] = after_dataset
    if base_model:
        body["base_model"] = base_model
    if trainer:
        body["trainer"] = trainer
    if total_steps:
        body["total_steps"] = int(total_steps)
    if config:
        body["config"] = dict(config)
    call = transport or _call
    created = call("POST", "/runs", api_key, body)
    run = TrainingRun(
        str(created["runId"]),
        name=name,
        api_key=api_key,
        total_steps=int(total_steps) if total_steps else None,
        flush_every=flush_every,
        flush_seconds=flush_seconds,
        transport=transport,
    )
    log.info("training run %s: %s", run.run_id, run.url)
    return run


def list_runs(*, api_key: str | None = None) -> list[dict[str, Any]]:
    out = _call("GET", "/runs", api_key)
    return list(out.get("runs") or []) if isinstance(out, dict) else []


def get_run(run_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    """The run plus ``series``: its points, oldest first."""
    return _call("GET", f"/runs/{run_id}", api_key)


def delete_run(run_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    return _call("DELETE", f"/runs/{run_id}", api_key)


def attach_delta(
    run_id: str,
    before: Sequence[dict],
    after: Sequence[dict],
    *,
    target: str | None = "pass_at_1",
    must_not_regress: Sequence[str] = (),
    api_key: str | None = None,
) -> dict[str, Any]:
    """Compute ``delta_report`` for a finished run and put it on the run
    page: the summary is re-sent with ``delta`` added, status unchanged."""
    from .score.delta import delta_report

    run = _call("GET", f"/runs/{run_id}", api_key)
    report = delta_report(before, after, target=target, must_not_regress=list(must_not_regress))
    summary = dict(run.get("summary") or {})
    summary["delta"] = _json_safe(report)
    status = str(run.get("status") or "done")
    if status == "running":
        status = "done"
    _call("POST", f"/runs/{run_id}/finish", api_key, {"status": status, "summary": summary})
    return report


# ---------------------------------------------------------------- Transformers / TRL


_EVENTS = (
    "on_init_end",
    "on_train_begin",
    "on_train_end",
    "on_epoch_begin",
    "on_epoch_end",
    "on_step_begin",
    "on_substep_end",
    "on_step_end",
    "on_optimizer_step",
    "on_pre_optimizer_step",
    "on_evaluate",
    "on_predict",
    "on_save",
    "on_log",
    "on_prediction_step",
)


class _NoOpCallback:
    """Stand-in base when transformers is not installed: every event the
    Trainer fires is a no-op, so the duck-typed callback still fits."""


for _event in _EVENTS:
    setattr(_NoOpCallback, _event, lambda self, *a, **k: None)


def _callback_base() -> type:
    try:
        from transformers import TrainerCallback as HFCallback

        return HFCallback
    except Exception:  # transformers is not a dependency of this package
        return _NoOpCallback


# Timing and throughput keys the Trainer logs alongside eval metrics; not
# learning signal, so not drawn.
_SKIP_KEYS = {
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "total_flos",
    "step",
}

_LOG_KEYS = {
    "loss": "loss",
    "eval_loss": "eval_loss",
    "learning_rate": "lr",
    "epoch": "epoch",
    "grad_norm": "grad_norm",
    "train_loss": "train_loss",
    "mean_token_accuracy": "token_accuracy",
    "eval_mean_token_accuracy": "eval_token_accuracy",
    "num_tokens": "tokens",
    # TRL RL trainers (GRPO, PPO, RLOO, online DPO): the curves an RL run is
    # read by. Any ``rewards/<name>`` key is kept under ``reward_<name>``.
    "reward": "reward",
    "reward_std": "reward_std",
    "kl": "kl",
    "objective/kl": "kl",
    "objective/rlhf_reward": "reward",
    "objective/scores": "score",
    "objective/entropy": "entropy",
    "entropy": "entropy",
    "completion_length": "completion_length",
    "completions/mean_length": "completion_length",
    "clip_ratio": "clip_ratio",
    "policy_loss": "policy_loss",
    "value_loss": "value_loss",
}


class TrainerCallback(_callback_base()):  # type: ignore[misc]
    """One line on a Transformers or TRL trainer:
    ``trainer.add_callback(zps.TrainerCallback(run))``.

    Logs every ``on_log`` to the run (loss, lr, eval loss, epoch, grad
    norm, token accuracy), takes the step count from the trainer at
    ``on_train_begin``, and finishes the run at ``on_train_end``. If the
    trainer raises, finish the run yourself with ``run.fail(...)`` or use
    the run as a context manager around ``trainer.train()``.
    """

    def __init__(self, run: TrainingRun):
        super().__init__()
        self.run = run

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        total = getattr(state, "max_steps", None)
        if total:
            self.run.progress(int(getattr(state, "global_step", 0) or 0), int(total))
        return control

    def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
        if not isinstance(logs, dict):
            return control
        metrics = {}
        for key, value in logs.items():
            if key in _SKIP_KEYS:
                continue
            name = _LOG_KEYS.get(key)
            if name is None and key.startswith("rewards/"):
                name = "reward_" + key[len("rewards/") :].replace("/", "_")
            if name is None and key.startswith("eval_") and isinstance(value, (int, float)):
                name = key
            if name is not None:
                metrics[name] = value
        step = int(getattr(state, "global_step", 0) or 0)
        if metrics:
            self.run.log(step, **metrics)
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        if self.run.status == "running":
            summary: dict[str, Any] = {}
            history = getattr(state, "log_history", None) or []
            for entry in reversed(history):
                if isinstance(entry, dict) and "train_loss" in entry:
                    summary["train_loss"] = entry["train_loss"]
                    break
            self.run.finish("done", summary=summary or None)
        return control


__all__ = [
    "TrainerCallback",
    "TrainingRun",
    "attach_delta",
    "delete_run",
    "get_run",
    "list_runs",
    "training_run",
]
