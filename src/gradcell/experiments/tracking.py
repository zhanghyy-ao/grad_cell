from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import traceback
import uuid
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "torch", "pybamm"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


class ExperimentRun:
    """Durable, append-only process and result recording for one experiment run."""

    def __init__(
        self,
        experiment: str,
        arguments: Namespace | dict[str, Any],
        root: str | Path = "results/runs",
        run_dir: str | Path | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}_{experiment}_{uuid.uuid4().hex[:8]}"
        self.experiment = experiment
        self.run_id = run_id
        self.path = Path(run_dir) if run_dir is not None else Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.events_path = self.path / "events.jsonl"
        self.log_path = self.path / "run.log"
        self.summary_path = self.path / "summary.json"
        self.started_monotonic = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).astimezone().isoformat()
        self.arguments = vars(arguments) if isinstance(arguments, Namespace) else arguments
        self.metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "experiment": experiment,
            "status": "running",
            "started_at": self.started_at,
            "command": [sys.executable, *sys.argv],
            "arguments": self.arguments,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "packages": _package_versions(),
            },
            "git": {
                "commit": _git_value("rev-parse", "HEAD"),
                "branch": _git_value("branch", "--show-current"),
                "dirty": bool(_git_value("status", "--porcelain")),
            },
        }
        _write_json(self.path / "metadata.json", self.metadata)
        self.event("run_started", run_dir=str(self.path))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, _traceback) -> bool:
        if exc is None:
            self.finish("completed")
        else:
            self.event(
                "run_failed",
                error=f"{exc_type.__name__}: {exc}",
                traceback="".join(traceback.format_exception(exc_type, exc, _traceback)),
            )
            self.finish("failed", error=f"{exc_type.__name__}: {exc}")
        return False

    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {level} | {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def event(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "elapsed_s": time.perf_counter() - self.started_monotonic,
            "event": event,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            handle.flush()

    def save_summary(self, payload: dict[str, Any]) -> None:
        summary = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "elapsed_s": time.perf_counter() - self.started_monotonic,
            **payload,
        }
        _write_json(self.summary_path, summary)

    def finish(self, status: str, **payload: Any) -> None:
        elapsed = time.perf_counter() - self.started_monotonic
        self.metadata.update(
            {
                "status": status,
                "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "elapsed_s": elapsed,
                **payload,
            }
        )
        _write_json(self.path / "metadata.json", self.metadata)
        self.event("run_finished", status=status, elapsed_s=elapsed)
        self.log(f"run={self.run_id} status={status} elapsed_s={elapsed:.2f}")
