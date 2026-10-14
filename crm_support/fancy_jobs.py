"""Persist web-run history and run one experiment at a time off the UI thread."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]


class JobStore:
    def __init__(self, directory=None):
        self.directory = Path(directory or ROOT / "outputs" / "fancy")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.process = None
        self.active_id = None
        self.log_file = None

    def submit(self, task):
        task = task.strip()
        if not task or len(task) > 4000:
            raise ValueError("Describe a task using between 1 and 4,000 characters.")
        with self.lock:
            self._reap()
            if self.process is not None:
                raise ValueError("An experiment is already running. Wait for it to finish.")
            job_id = uuid.uuid4().hex
            directory = self.directory / job_id
            directory.mkdir()
            (directory / "task.txt").write_text(task, encoding="utf-8")
            (directory / "job.json").write_text(json.dumps({
                "id": job_id, "task": task,
                "created": datetime.now(timezone.utc).isoformat(),
            }), encoding="utf-8")
            self.log_file = (directory / "run.log").open("w", encoding="utf-8")
            try:
                self.process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "scripts.fancy_run", str(directory.resolve())],
                    cwd=ROOT, stdout=self.log_file, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYGAME_HIDE_SUPPORT_PROMPT": "1"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception:
                self.log_file.close()
                self.log_file = None
                raise
            self.active_id = job_id
            return job_id

    def _reap(self):
        if self.process is not None and self.process.poll() is not None:
            self.log_file.close()
            self.log_file = None
            self.process = None
            self.active_id = None

    def snapshot(self):
        with self.lock:
            self._reap()
            rows = []
            for path in self.directory.glob("*/job.json"):
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                    result_path = path.parent / "result.json"
                    if result_path.exists():
                        job.update(json.loads(result_path.read_text(encoding="utf-8")))
                    else:
                        job["status"] = "running" if job["id"] == self.active_id else "interrupted"
                    job["directory"] = str(path.parent)
                    rows.append(job)
                except (OSError, ValueError):
                    continue
            return sorted(rows, key=lambda row: row["created"], reverse=True)


def progress(directory):
    path = Path(directory) / "run.log"
    if not path.exists():
        return "Starting the experiment…"
    with path.open("rb") as log:
        log.seek(max(0, path.stat().st_size - 12000))
        lines = log.read().decode("utf-8", errors="replace").splitlines()
    prefixes = ("Generating", "Training", "Validating", "Correcting", "Evaluating",
                "Judging", "Preparing", "Iteration", "Judge is")
    return next((line for line in reversed(lines) if line.startswith(prefixes)),
                "Starting the experiment…")
