from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
REPO = WORKSPACE / "baseline" / "SimpleMem"
CONFIG = REPO / "config.py"
DEFAULT_DATASET = WORKSPACE / "datasets" / "locomo" / "data" / "locomo10.json"
OUTPUT_ROOT = WORKSPACE / "runs" / "simplemem"

PRESET_ARGS = {
    "smoke5": ["--num-samples", "5"],
    "smoke20": ["--num-samples", "20"],
    "full": [],
    "judge20": ["--num-samples", "20", "--llm-judge"],
    "judgefull": ["--llm-judge"],
}


def redact_config(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    text = text.replace('OPENAI_API_KEY = "REPLACE_WITH_YOUR_API_KEY"', 'OPENAI_API_KEY = "***REDACTED***"')
    text = text.replace('JUDGE_API_KEY = OPENAI_API_KEY', 'JUDGE_API_KEY = "***REDACTED_OR_MAIN_KEY***"')
    for prefix in ('OPENAI_API_KEY = "', 'JUDGE_API_KEY = "'):
        lines = []
        for line in text.splitlines():
            if line.startswith(prefix):
                lines.append(prefix + "***REDACTED***\"")
            else:
                lines.append(line)
        text = "\n".join(lines)
    dst.write_text(text + "\n", encoding="utf-8")


def git_commit(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def stream_process(cmd: list[str], cwd: Path, log_file: Path) -> int:
    state = {
        "last_output_at": time.time(),
        "heartbeat_at": 0.0,
    }
    lock = threading.Lock()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None

        def reader() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
                with lock:
                    state["last_output_at"] = time.time()

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        while process.poll() is None:
            time.sleep(5)
            now = time.time()
            with lock:
                silent_for = now - state["last_output_at"]
                can_heartbeat = now - state["heartbeat_at"] >= 20
                if silent_for >= 20 and can_heartbeat:
                    state["heartbeat_at"] = now
                    print(
                        f"[progress] still running... no new log for {int(silent_for)}s. "
                        f"First run may be downloading NLTK data or HuggingFace models. "
                        f"Live log: {log_file}"
                    )

        reader_thread.join(timeout=2)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SimpleMem LoCoMo experiments with saved outputs.")
    parser.add_argument(
        "preset",
        choices=sorted(PRESET_ARGS.keys()),
        help="smoke5 / smoke20 / full / judge20 / judgefull",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to locomo10.json")
    parser.add_argument("--python", default=sys.executable, help="Python executable with SimpleMem dependencies installed")
    parser.add_argument("--parallel-questions", action="store_true", help="Pass through to test_locomo10.py")
    parser.add_argument("--test-workers", type=int, default=0, help="Pass through to test_locomo10.py")
    parser.add_argument("--output-dir", default="", help="Optional fixed output directory for this run")
    args = parser.parse_args()

    if not REPO.exists():
        parser.error(f"SimpleMem repo not found at {REPO}")
    if not CONFIG.exists():
        parser.error(f"Config not found at {CONFIG}")

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        parser.error(f"Dataset file not found: {dataset}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_ROOT / f"{args.preset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result_file = run_dir / "result.json"
    log_file = run_dir / "run.log"
    meta_file = run_dir / "meta.txt"
    config_file = run_dir / "config_redacted.py"
    command_file = run_dir / "command.txt"

    cmd = [args.python, "-u", "test_locomo10.py", "--dataset", str(dataset), "--result-file", str(result_file)]
    cmd.extend(PRESET_ARGS[args.preset])
    if args.parallel_questions:
        cmd.append("--parallel-questions")
    if args.test_workers > 0:
        cmd.extend(["--test-workers", str(args.test_workers)])

    commit = git_commit(REPO)
    meta_file.write_text(
        "\n".join(
            [
                f"preset={args.preset}",
                f"started_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"repo={REPO}",
                f"dataset={dataset}",
                f"commit={commit}",
                f"python={args.python}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    command_file.write_text(" ".join(f'"{part}"' if " " in part else part for part in cmd) + "\n", encoding="utf-8")
    redact_config(CONFIG, config_file)

    print(f"Running preset: {args.preset}")
    print(f"Dataset: {dataset}")
    print(f"Results: {run_dir}")
    print(f"Python: {args.python}")
    print(f"Commit: {commit}")
    print("Progress notes:")
    print("- First run may spend a few minutes downloading NLTK resources and HuggingFace models.")
    print("- In local Transformers mode, the first run will also download the Qwen2.5-1.5B model weights.")
    print("- If SimpleMem itself is quiet for 20 seconds, this wrapper will print a heartbeat line.")
    print("- Full LoCoMo is much slower than smoke5/smoke20; use smoke5 first when debugging.")

    exit_code = stream_process(cmd, REPO, log_file)
    if exit_code != 0:
        print(f"\nRun failed with exit code {exit_code}. See {log_file}")
        return exit_code

    print(f"\nRun finished successfully. Result file: {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
