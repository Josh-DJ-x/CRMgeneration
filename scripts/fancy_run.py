"""Isolated experiment worker used by the local web interface."""

import json
import os
from pathlib import Path
import sys


def run(job_path):
    job_path = Path(job_path).resolve()
    task = (job_path / "task.txt").read_text(encoding="utf-8")
    os.environ["GRIDWORLD_ANIMATION_PATH"] = str(job_path / "animation.gif")
    os.environ["GRIDWORLD_PREVIEW_PATH"] = str(job_path / "grid.json")
    try:
        from system import JUDGE_LLM, JUDGE_PROVIDER
        from system_loop import main
        record = main(task_description=task, iterative=True, long=True,
                      generator_backend="gemini", generator_model="gemini-3.5-flash-lite",
                      judge_backend=JUDGE_PROVIDER,
                      judge_model=JUDGE_LLM if JUDGE_PROVIDER == "openrouter" else "gpt-4.1")
        result = {"status": "completed", "record": record}
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        result = {"status": "error", "error": str(error) or type(error).__name__}
    pending = job_path / "result.tmp"
    pending.write_text(json.dumps(result, default=str), encoding="utf-8")
    pending.replace(job_path / "result.json")


if __name__ == "__main__":
    run(sys.argv[1])
