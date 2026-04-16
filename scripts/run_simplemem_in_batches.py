from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = WORKSPACE / "datasets" / "locomo" / "data" / "locomo10.json"
DEFAULT_JOBS_ROOT = WORKSPACE / "runs" / "simplemem" / "batch_jobs"
RUN_SIMPLEMEM = WORKSPACE / "run_simplemem.py"
MERGE_SCRIPT = WORKSPACE / "scripts" / "merge_simplemem_results.py"


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Dataset must be a JSON list, got: {type(obj)}")
    return obj


def _split_batches(total: int, batch_size: int) -> list[tuple[int, int]]:
    batches: list[tuple[int, int]] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batches.append((start, end))
    return batches


def _write_batch_datasets(
    data: list[dict[str, Any]],
    batches: list[tuple[int, int]],
    datasets_dir: Path,
) -> list[Path]:
    datasets_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (start, end) in enumerate(batches, start=1):
        p = datasets_dir / f"batch_{i:02d}.json"
        p.write_text(json.dumps(data[start:end], ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(p)
    return paths


def _write_manifest(
    manifest_path: Path,
    dataset: Path,
    batch_size: int,
    total_samples: int,
    batch_paths: list[Path],
    batches: list[tuple[int, int]],
) -> None:
    payload = {
        "dataset": str(dataset),
        "batch_size": batch_size,
        "total_samples": total_samples,
        "total_batches": len(batches),
        "batches": [],
    }
    for i, ((start, end), batch_path) in enumerate(zip(batches, batch_paths), start=1):
        payload["batches"].append(
            {
                "batch_id": i,
                "source_sample_start": start,
                "source_sample_end_exclusive": end,
                "num_samples": end - start,
                "dataset_file": str(batch_path),
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_one_batch(
    python_exe: str,
    batch_dataset: Path,
    batch_output_dir: Path,
    parallel_questions: bool,
    test_workers: int,
) -> int:
    cmd = [
        python_exe,
        str(RUN_SIMPLEMEM),
        "full",
        "--dataset",
        str(batch_dataset),
        "--python",
        python_exe,
        "--output-dir",
        str(batch_output_dir),
    ]
    if parallel_questions:
        cmd.append("--parallel-questions")
    if test_workers > 0:
        cmd.extend(["--test-workers", str(test_workers)])

    print(f"\n[batch] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(WORKSPACE))
    return proc.returncode


def _merge_all_results(python_exe: str, result_files: list[Path], output_file: Path) -> int:
    cmd = [python_exe, str(MERGE_SCRIPT), "--inputs", *[str(x) for x in result_files], "--output", str(output_file)]
    print(f"\n[merge] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(WORKSPACE))
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SimpleMem in fixed-size sample batches and merge results automatically."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to source dataset json.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of samples per batch (default: 2).")
    parser.add_argument("--start-batch", type=int, default=1, help="1-based batch index to start from.")
    parser.add_argument("--end-batch", type=int, default=0, help="1-based batch index to stop at (0 means last batch).")
    parser.add_argument(
        "--job-name",
        default="locomo10_batch2",
        help="Job folder name under runs/simplemem/batch_jobs, reused for resume.",
    )
    parser.add_argument(
        "--jobs-root",
        default=str(DEFAULT_JOBS_ROOT),
        help="Root directory to store batch jobs (use a Google Drive path on Colab for persistence).",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to run child scripts.")
    parser.add_argument("--parallel-questions", action="store_true", help="Pass through to run_simplemem.py.")
    parser.add_argument("--test-workers", type=int, default=0, help="Pass through to run_simplemem.py.")
    parser.add_argument("--force-rerun", action="store_true", help="Rerun batch even if result.json already exists.")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        parser.error(f"Dataset file not found: {dataset}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.start_batch <= 0:
        parser.error("--start-batch must be >= 1")

    data = _load_dataset(dataset)
    total_samples = len(data)
    if total_samples == 0:
        parser.error("Dataset is empty")

    batches = _split_batches(total_samples, args.batch_size)
    total_batches = len(batches)
    end_batch = args.end_batch if args.end_batch > 0 else total_batches

    if args.start_batch > total_batches:
        parser.error(f"--start-batch ({args.start_batch}) exceeds total batches ({total_batches})")
    if end_batch > total_batches:
        parser.error(f"--end-batch ({end_batch}) exceeds total batches ({total_batches})")
    if end_batch < args.start_batch:
        parser.error("--end-batch must be >= --start-batch")

    jobs_root = Path(args.jobs_root).resolve()
    job_root = jobs_root / args.job_name
    datasets_dir = job_root / "datasets"
    manifest_path = job_root / "manifest.json"

    batch_dataset_paths = _write_batch_datasets(data, batches, datasets_dir)
    _write_manifest(manifest_path, dataset, args.batch_size, total_samples, batch_dataset_paths, batches)

    print(f"job root: {job_root}")
    print(f"dataset: {dataset}")
    print(f"total samples: {total_samples}")
    print(f"batch size: {args.batch_size}")
    print(f"total batches: {total_batches}")
    print(f"run range: {args.start_batch}..{end_batch}")

    for batch_id in range(args.start_batch, end_batch + 1):
        batch_dataset = batch_dataset_paths[batch_id - 1]
        src_start, src_end = batches[batch_id - 1]
        batch_output_dir = job_root / f"batch_{batch_id:02d}"
        result_file = batch_output_dir / "result.json"

        print(
            f"\n=== Batch {batch_id:02d}/{total_batches} | "
            f"source sample [{src_start}, {src_end}) | output: {batch_output_dir} ==="
        )

        if result_file.exists() and not args.force_rerun:
            print(f"[batch] skip existing result: {result_file}")
            continue

        code = _run_one_batch(
            python_exe=args.python,
            batch_dataset=batch_dataset,
            batch_output_dir=batch_output_dir,
            parallel_questions=args.parallel_questions,
            test_workers=args.test_workers,
        )
        if code != 0:
            print(f"\n[batch] failed at batch {batch_id:02d} (exit={code})")
            print(
                f"[batch] resume with: {args.python} {Path(__file__).name} "
                f"--job-name {args.job_name} --start-batch {batch_id}"
            )
            return code

    expected_results = [job_root / f"batch_{i:02d}" / "result.json" for i in range(1, total_batches + 1)]
    missing = [p for p in expected_results if not p.exists()]
    if missing:
        print("\n[merge] skipped because some batch results are missing:")
        for p in missing:
            print(f"  - {p}")
        print("[merge] complete remaining batches, then rerun this script with --start-batch pointing to the first missing batch.")
        return 0

    merged_output = job_root / "merged" / "result.json"
    merge_code = _merge_all_results(args.python, expected_results, merged_output)
    if merge_code != 0:
        print(f"[merge] failed (exit={merge_code})")
        return merge_code

    print(f"\n[done] merged result: {merged_output}")
    print(f"[done] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
