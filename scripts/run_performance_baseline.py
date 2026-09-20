"""Run the protected local performance benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from knowledge_rag.benchmark_corpus import load_benchmark_workload, workload_to_dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local performance baseline.")
    parser.add_argument("--allow-reset", action="store_true")
    parser.add_argument("--workload", type=Path, default=REPOSITORY_ROOT / "benchmarks/workload.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "benchmarks/results/latest.json")
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--diagnostic-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = _load_settings()
    except Exception as exc:  # noqa: BLE001 - configuration details may contain credentials.
        return _report_failure(args, "target_validation", type(exc).__name__)
    try:
        config = _benchmark_module().BenchmarkRunConfig(
            repository_root=REPOSITORY_ROOT,
            benchmark_database_url=settings.benchmark_database_url or "",
            application_database_url=settings.database_url,
            workload_path=args.workload,
            sample_vault_path=REPOSITORY_ROOT / "examples/sample-vault",
            evaluation_dataset_path=REPOSITORY_ROOT / "evaluation/retrieval_cases.json",
            thresholds_path=REPOSITORY_ROOT / "evaluation/thresholds.json",
            allow_reset=args.allow_reset,
            work_directory=args.work_directory,
        )
    except Exception as exc:  # noqa: BLE001 - deferred imports may load malformed settings.
        return _report_failure(args, "target_validation", type(exc).__name__)
    try:
        report = run_performance_baseline(config)
    except Exception as error:  # noqa: BLE001 - runner failures must remain public-safe.
        return _report_failure(
            args,
            getattr(error, "stage", "target_validation"),
            getattr(error, "error_type", type(error).__name__),
        )
    try:
        write_benchmark_report(report, args.output, REPOSITORY_ROOT / "benchmarks/baselines")
    except Exception as exc:  # noqa: BLE001 - command output must never expose exception text.
        return _report_failure(args, "output_write", type(exc).__name__)
    print(format_benchmark_summary(report))
    return 0


def _load_settings():
    from knowledge_rag.config import Settings

    return Settings()


def _benchmark_module():
    from knowledge_rag import performance_benchmark

    return performance_benchmark


def run_performance_baseline(config):
    return _benchmark_module().run_performance_baseline(config)


def write_benchmark_report(report, output_path: Path, baselines_dir: Path) -> None:
    _benchmark_module().write_benchmark_report(report, output_path, baselines_dir)


def format_benchmark_summary(report) -> str:
    return _benchmark_module().format_benchmark_summary(report)


def make_incomplete_report(workload_version, workload, stage: str, error_type: str):
    return _benchmark_module().make_incomplete_report(workload_version, workload, stage, error_type)


def _report_failure(args: argparse.Namespace, stage: str, error_type: str) -> int:
    print(f"benchmark incomplete: {stage}: {error_type}")
    if args.diagnostic_output is not None:
        try:
            workload = load_benchmark_workload(args.workload)
            diagnostic = make_incomplete_report(
                workload.version, workload_to_dict(workload), stage, error_type
            )
            write_benchmark_report(
                diagnostic, args.diagnostic_output, REPOSITORY_ROOT / "benchmarks/baselines"
            )
        except Exception:  # noqa: BLE001 - a diagnostic write must not leak sensitive details.
            return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
