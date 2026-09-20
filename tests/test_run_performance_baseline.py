from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@contextmanager
def _directory_redirect(link: Path, target: Path):
    """Create a directory redirect without requiring Windows symlink privilege."""

    is_symlink = True
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("directory symlinks and junctions are unavailable")
        is_symlink = False

    try:
        yield
    finally:
        if is_symlink:
            link.unlink()
        elif link.exists():
            # rmdir removes a junction itself; it does not recurse into its target.
            link.rmdir()
        assert target.is_dir()


def _complete_report():
    from knowledge_rag.benchmark_models import BenchmarkReport, EnvironmentMetadata

    return BenchmarkReport(
        complete=True, schema_version=1, workload_version=1, git_commit=None,
        executed_at_utc="2026-09-18T00:00:00Z",
        environment=EnvironmentMetadata("3.12", "Linux", 1, 1, "16", "0.8"),
        workload={"version": 1, "seed": 14069, "tiers": [100, 1000, 10000], "chunks_per_note": 3, "changed_fraction": 0.1, "warmup_iterations": 5, "measured_iterations": 30, "queries": []},
        quality=None, tiers=(),
    )


def test_orchestrator_validates_before_connects_and_records_failed_quality(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark
    from knowledge_rag.benchmark_corpus import CorpusManifest, load_benchmark_workload
    from knowledge_rag.benchmark_models import QualityBaselineResult

    workload = load_benchmark_workload(Path("benchmarks/workload.json"))
    events: list[str] = []
    connection = object()
    monkeypatch.setattr(benchmark, "validate_benchmark_target", lambda *args: events.append("validate"))
    monkeypatch.setattr(benchmark.psycopg, "connect", lambda url: events.append("connect") or nullcontext(connection))
    monkeypatch.setattr(benchmark, "reset_benchmark_tables", lambda conn: events.append("reset"))
    monkeypatch.setattr(benchmark, "load_benchmark_workload", lambda path: workload)
    monkeypatch.setattr(
        benchmark,
        "run_indexing_stage",
        lambda *args, **kwargs: events.append("index") or SimpleNamespace(observed_chunks=3),
    )
    monkeypatch.setattr(benchmark, "run_embedding_stage", lambda *args, **kwargs: events.append("embed") or object())
    monkeypatch.setattr(
        benchmark,
        "run_quality_baseline",
        lambda **kwargs: events.append("quality") or QualityBaselineResult(False, {"cases": [], "summaries": []}, ()),
    )
    monkeypatch.setattr(
        benchmark,
        "generate_synthetic_vault",
        lambda root, loaded, count: events.append(f"generate:{count}") or CorpusManifest(1, 1, count, count * 3, (), "digest"),
    )
    monkeypatch.setattr(benchmark, "modify_synthetic_vault", lambda *args: events.append("mutate") or ())
    monkeypatch.setattr(benchmark, "run_retrieval_latency", lambda **kwargs: events.append("retrieve") or ((), ()))
    monkeypatch.setattr(benchmark, "collect_environment_metadata", lambda conn: events.append("environment") or None)
    monkeypatch.setattr(benchmark, "_git_commit", lambda root: "abc1234")

    config = benchmark.BenchmarkRunConfig(
        repository_root=tmp_path,
        benchmark_database_url="postgresql://benchmark",
        application_database_url="postgresql://application",
        workload_path=tmp_path / "workload.json",
        sample_vault_path=tmp_path / "sample",
        evaluation_dataset_path=tmp_path / "evaluation.json",
        thresholds_path=tmp_path / "thresholds.json",
        allow_reset=True,
        work_directory=tmp_path / "work",
    )
    preserved = tmp_path / "work" / "pre-existing.txt"
    # The supplied directory is a caller-owned parent, never a disposable vault.
    (tmp_path / "work").mkdir()
    preserved.write_text("keep", encoding="utf-8")
    report = benchmark.run_performance_baseline(config)

    assert events == [
        "validate", "connect", "reset", "index", "embed", "quality",
        *(
            ["connect", "reset", "generate:100", "index", "embed", "index", "mutate", "index", "embed", "retrieve"]
            + ["connect", "reset", "generate:1000", "index", "embed", "index", "mutate", "index", "embed", "retrieve"]
            + ["connect", "reset", "generate:10000", "index", "embed", "index", "mutate", "index", "embed", "retrieve"]
        ),
        "connect", "environment",
    ]
    assert report.complete is True
    assert report.quality is not None and report.quality.passed is False
    assert [tier.note_count for tier in report.tiers] == [100, 1000, 10000]
    assert preserved.read_text(encoding="utf-8") == "keep"
    assert list((tmp_path / "work").iterdir()) == [preserved]


def test_orchestrator_sanitizes_stage_errors_and_stops(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark
    from knowledge_rag.benchmark_corpus import load_benchmark_workload

    workload = load_benchmark_workload(Path("benchmarks/workload.json"))
    monkeypatch.setattr(benchmark, "validate_benchmark_target", lambda *args: None)
    monkeypatch.setattr(benchmark, "load_benchmark_workload", lambda path: workload)
    monkeypatch.setattr(benchmark.psycopg, "connect", lambda url: nullcontext(object()))
    monkeypatch.setattr(benchmark, "reset_benchmark_tables", lambda conn: (_ for _ in ()).throw(RuntimeError("postgresql://secret@host/private")))

    config = benchmark.BenchmarkRunConfig(tmp_path, "benchmark", "application", tmp_path / "workload", tmp_path / "sample", tmp_path / "eval", tmp_path / "thresholds", True)
    with pytest.raises(benchmark.BenchmarkStageError) as error:
        benchmark.run_performance_baseline(config)

    assert error.value.stage == "database_reset"
    assert str(error.value) == "database_reset: RuntimeError"
    assert "secret" not in str(error.value)


def test_orchestrator_preserves_populated_work_directory_when_reset_fails(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark
    from knowledge_rag.benchmark_corpus import load_benchmark_workload

    workload = load_benchmark_workload(Path("benchmarks/workload.json"))
    work = tmp_path / "work"
    work.mkdir()
    preserved = work / "tier-100"
    preserved.mkdir()
    marker = preserved / "keep.txt"
    marker.write_text("do not remove", encoding="utf-8")
    monkeypatch.setattr(benchmark, "validate_benchmark_target", lambda *args: None)
    monkeypatch.setattr(benchmark, "load_benchmark_workload", lambda path: workload)
    monkeypatch.setattr(benchmark.psycopg, "connect", lambda url: nullcontext(object()))
    monkeypatch.setattr(benchmark, "reset_benchmark_tables", lambda conn: (_ for _ in ()).throw(RuntimeError("secret reset")))
    monkeypatch.setattr(benchmark, "generate_synthetic_vault", lambda *args: pytest.fail("must not generate"))

    config = benchmark.BenchmarkRunConfig(tmp_path, "benchmark", "application", tmp_path / "workload", tmp_path / "sample", tmp_path / "eval", tmp_path / "thresholds", True, work)
    with pytest.raises(benchmark.BenchmarkStageError, match=r"^database_reset: RuntimeError$"):
        benchmark.run_performance_baseline(config)
    assert marker.read_text(encoding="utf-8") == "do not remove"


def test_context_entry_and_exit_failures_are_sanitized(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark
    from knowledge_rag.benchmark_corpus import load_benchmark_workload

    class ExplodingConnection:
        def __enter__(self):
            raise RuntimeError("postgresql://secret@private")

        def __exit__(self, *args):
            raise RuntimeError("private exit")

    workload = load_benchmark_workload(Path("benchmarks/workload.json"))
    monkeypatch.setattr(benchmark, "validate_benchmark_target", lambda *args: None)
    monkeykey = monkeypatch.setattr
    monkeykey(benchmark, "load_benchmark_workload", lambda path: workload)
    monkeykey(benchmark.psycopg, "connect", lambda url: ExplodingConnection())
    config = benchmark.BenchmarkRunConfig(tmp_path, "benchmark", "application", tmp_path / "workload", tmp_path / "sample", tmp_path / "eval", tmp_path / "thresholds", True)

    with pytest.raises(benchmark.BenchmarkStageError, match=r"^database_reset: RuntimeError$") as error:
        benchmark.run_performance_baseline(config)
    assert "secret" not in str(error.value)


def test_cleanup_failure_preserves_primary_stage_failure(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark
    from knowledge_rag.benchmark_corpus import CorpusManifest, load_benchmark_workload
    from knowledge_rag.benchmark_models import QualityBaselineResult

    workload = load_benchmark_workload(Path("benchmarks/workload.json"))
    calls = 0
    monkeypatch.setattr(benchmark, "validate_benchmark_target", lambda *args: None)
    monkeypatch.setattr(benchmark, "load_benchmark_workload", lambda path: workload)
    monkeypatch.setattr(benchmark.psycopg, "connect", lambda url: nullcontext(object()))
    monkeypatch.setattr(benchmark, "reset_benchmark_tables", lambda conn: None)
    monkeypatch.setattr(benchmark, "run_embedding_stage", lambda *args: object())
    monkeypatch.setattr(benchmark, "run_quality_baseline", lambda **kwargs: QualityBaselineResult(True, {"cases": [], "summaries": []}, ()))
    monkeypatch.setattr(benchmark, "generate_synthetic_vault", lambda root, *args: CorpusManifest(1, 1, 100, 300, (), "digest"))

    def fail_first_tier(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("secret indexing failure")
        return SimpleNamespace(observed_chunks=3)

    monkeypatch.setattr(benchmark, "run_indexing_stage", fail_first_tier)
    monkeypatch.setattr(benchmark, "rmtree", lambda path: (_ for _ in ()).throw(RuntimeError("secret cleanup")))
    config = benchmark.BenchmarkRunConfig(tmp_path, "benchmark", "application", tmp_path / "workload", tmp_path / "sample", tmp_path / "eval", tmp_path / "thresholds", True)

    with pytest.raises(benchmark.BenchmarkStageError, match=r"^clean_indexing: RuntimeError$") as error:
        benchmark.run_performance_baseline(config)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("field,value", [("tiers", (100,)), ("warmup_iterations", 4), ("measured_iterations", 29)])
def test_unapproved_workload_stops_before_connections_or_work(field, value, monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark
    from knowledge_rag.benchmark_corpus import load_benchmark_workload

    workload = replace(load_benchmark_workload(Path("benchmarks/workload.json")), **{field: value})
    events: list[str] = []
    monkeypatch.setattr(benchmark, "validate_benchmark_target", lambda *args: events.append("validate"))
    monkeypatch.setattr(benchmark, "load_benchmark_workload", lambda path: workload)
    monkeypatch.setattr(benchmark.psycopg, "connect", lambda url: events.append("connect"))
    config = benchmark.BenchmarkRunConfig(tmp_path, "benchmark", "application", tmp_path / "workload", tmp_path / "sample", tmp_path / "eval", tmp_path / "thresholds", True)

    with pytest.raises(benchmark.BenchmarkStageError, match=r"^target_validation: ValueError$"):
        benchmark.run_performance_baseline(config)
    assert events == ["validate"]


def test_report_writing_protects_canonical_baselines_and_is_stable(tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark

    workload = {
        "version": 1, "seed": 1, "tiers": [], "chunks_per_note": 3,
        "changed_fraction": 0.1, "warmup_iterations": 0, "measured_iterations": 1, "queries": [],
    }
    report = benchmark.make_incomplete_report(1, workload, "database_reset", "RuntimeError")
    baselines = tmp_path / "benchmarks" / "baselines"
    with pytest.raises(ValueError, match="incomplete report cannot replace a canonical baseline"):
        benchmark.write_benchmark_report(report, baselines / "baseline.json", baselines)

    output = tmp_path / "diagnostics" / "failure.json"
    benchmark.write_benchmark_report(report, output, baselines)
    serialized = output.read_text(encoding="utf-8")
    assert serialized.endswith("\n")
    assert json.loads(serialized)["failure_stage"] == "database_reset"


def test_complete_canonical_output_is_sorted_indented_and_atomic(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark

    output = tmp_path / "benchmarks" / "baselines" / "baseline.json"
    output.parent.mkdir(parents=True)
    output.write_text("old complete report\n", encoding="utf-8")
    benchmark.write_benchmark_report(_complete_report(), output, output.parent)
    serialized = output.read_text(encoding="utf-8")
    assert serialized.startswith('{\n  "complete": true,')
    assert serialized.endswith("\n")
    assert list(json.loads(serialized)) == sorted(json.loads(serialized))

    monkeypatch.setattr(Path, "replace", lambda self, target: (_ for _ in ()).throw(OSError("private replace")))
    with pytest.raises(OSError, match="private replace"):
        benchmark.write_benchmark_report(_complete_report(), output, output.parent)
    assert output.read_text(encoding="utf-8") == serialized


def test_complete_canonical_output_has_exact_json_representation(tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark

    output = tmp_path / "benchmarks" / "baselines" / "baseline.json"
    benchmark.write_benchmark_report(_complete_report(), output, output.parent)

    assert output.read_text(encoding="utf-8") == """{
  "complete": true,
  "environment": {
    "logical_cpu_count": 1,
    "operating_system": "Linux",
    "pgvector_version": "0.8",
    "postgresql_version": "16",
    "python_version": "3.12",
    "total_memory_bytes": 1
  },
  "executed_at_utc": "2026-09-18T00:00:00Z",
  "failure_stage": null,
  "failure_type": null,
  "git_commit": null,
  "quality": null,
  "schema_version": 1,
  "tiers": [],
  "workload": {
    "changed_fraction": 0.1,
    "chunks_per_note": 3,
    "measured_iterations": 30,
    "queries": [],
    "seed": 14069,
    "tiers": [
      100,
      1000,
      10000
    ],
    "version": 1,
    "warmup_iterations": 5
  },
  "workload_version": 1
}
"""


def test_summary_reports_tier_metrics_without_private_data() -> None:
    from knowledge_rag import performance_benchmark as benchmark
    from knowledge_rag.benchmark_models import (
        EmbeddingBenchmarkResult,
        IndexingBenchmarkResult,
        QualityBaselineResult,
        RateMeasurement,
        TierBenchmarkResult,
    )

    rate = RateMeasurement(300, 1_000_000_000, 300.0)
    indexing = IndexingBenchmarkResult("clean", 100, 100, 0, 0, 0, 0, 100, 300, 300, rate, rate)
    embedding = EmbeddingBenchmarkResult(300, 300, 300, 1, rate, rate, rate)
    tier = TierBenchmarkResult(100, 300, 300, indexing, embedding, indexing, indexing, embedding, (), ())
    report = replace(
        _complete_report(),
        quality=QualityBaselineResult(True, {"cases": [], "summaries": [{"mean_recall_at_k": 1.0}]}, ()),
        tiers=(tier,),
    )

    summary = benchmark.format_benchmark_summary(report)
    assert "tier 100: chunks=300" in summary
    assert "clean_chunks_per_second=300.00" in summary
    assert "embedding_per_second=300.00" in summary
    assert "quality_passed=true" in summary
    assert "postgresql://" not in summary
    assert "private note" not in summary


def test_summary_and_cli_defaults_are_public_and_stable() -> None:
    from knowledge_rag import performance_benchmark as benchmark
    from scripts import run_performance_baseline as command

    args = command.parse_args([])
    assert args.allow_reset is False
    assert args.workload == command.REPOSITORY_ROOT / "benchmarks/workload.json"
    assert args.output == command.REPOSITORY_ROOT / "benchmarks/results/latest.json"
    assert args.diagnostic_output is None
    summary = benchmark.format_benchmark_summary(_complete_report())
    assert "benchmark complete=true" in summary
    assert "quality_passed=not run" in summary
    assert "postgresql://" not in summary


def test_cli_defaults_and_allow_reset_gate_are_passed_to_runner(monkeypatch, capsys) -> None:
    from scripts import run_performance_baseline as command

    captured = []
    monkeypatch.setattr(command, "_load_settings", lambda: SimpleNamespace(benchmark_database_url="benchmark", database_url="application"))
    monkeypatch.setattr(command, "run_performance_baseline", lambda config: captured.append(config) or (_ for _ in ()).throw(RuntimeError("stop")))

    assert command.main([]) == 1
    assert captured[0].workload_path == command.REPOSITORY_ROOT / "benchmarks/workload.json"
    assert captured[0].work_directory is None
    assert captured[0].allow_reset is False
    assert command.main(["--allow-reset"]) == 1
    assert captured[1].allow_reset is True
    assert capsys.readouterr().out == "benchmark incomplete: target_validation: RuntimeError\n" * 2


def test_later_tier_failure_removes_only_run_owned_work_root(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark
    from knowledge_rag.benchmark_corpus import CorpusManifest, load_benchmark_workload
    from knowledge_rag.benchmark_models import QualityBaselineResult

    supplied = tmp_path / "supplied"
    supplied.mkdir()
    sentinel = supplied / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    workload = load_benchmark_workload(Path("benchmarks/workload.json"))
    monkeypatch.setattr(benchmark, "validate_benchmark_target", lambda *args: None)
    monkeypatch.setattr(benchmark, "load_benchmark_workload", lambda path: workload)
    monkeypatch.setattr(benchmark.psycopg, "connect", lambda url: nullcontext(object()))
    monkeypatch.setattr(benchmark, "reset_benchmark_tables", lambda conn: None)
    monkeypatch.setattr(benchmark, "run_embedding_stage", lambda *args: object())
    monkeypatch.setattr(benchmark, "run_quality_baseline", lambda **kwargs: QualityBaselineResult(True, {"cases": [], "summaries": []}, ()))

    def generate(root, _workload, count):
        root.mkdir()
        (root / "generated.md").write_text("generated", encoding="utf-8")
        return CorpusManifest(1, 14069, count, count * 3, ("generated.md",), "digest")

    def index(_conn, root, scenario, *args):
        if root.name == "tier-1000" and scenario == "clean":
            raise RuntimeError("later tier secret")
        return SimpleNamespace(observed_chunks=3)

    monkeypatch.setattr(benchmark, "generate_synthetic_vault", generate)
    monkeypatch.setattr(benchmark, "run_indexing_stage", index)
    monkeypatch.setattr(benchmark, "modify_synthetic_vault", lambda *args: ())
    monkeypatch.setattr(benchmark, "run_retrieval_latency", lambda **kwargs: ((), ()))
    config = benchmark.BenchmarkRunConfig(tmp_path, "benchmark", "application", tmp_path / "workload", tmp_path / "sample", tmp_path / "eval", tmp_path / "thresholds", True, supplied)

    with pytest.raises(benchmark.BenchmarkStageError, match=r"^clean_indexing: RuntimeError$"):
        benchmark.run_performance_baseline(config)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert list(supplied.iterdir()) == [sentinel]


def test_cli_import_and_main_are_safe_with_malformed_environment() -> None:
    environment = os.environ.copy()
    environment["EXCLUDED_PATHS"] = "[not valid json"
    imported = subprocess.run(
        [sys.executable, "-c", "import scripts.run_performance_baseline"],
        cwd=Path.cwd(), env=environment, capture_output=True, text=True, check=False,
    )
    assert imported.returncode == 0
    assert imported.stderr == ""

    executed = subprocess.run(
        [sys.executable, "-c", "from scripts.run_performance_baseline import main; raise SystemExit(main())"],
        cwd=Path.cwd(), env=environment, capture_output=True, text=True, check=False,
    )
    assert executed.returncode == 1
    assert executed.stdout == "benchmark incomplete: target_validation: SettingsError\n"
    assert "not valid" not in executed.stderr


def test_incomplete_report_rejects_lexically_canonical_symlink_output(tmp_path) -> None:
    from knowledge_rag import performance_benchmark as benchmark

    workload = {"version": 1, "seed": 1, "tiers": [], "chunks_per_note": 3, "changed_fraction": 0.1, "warmup_iterations": 0, "measured_iterations": 1, "queries": []}
    report = benchmark.make_incomplete_report(1, workload, "database_reset", "RuntimeError")
    baselines = tmp_path / "benchmarks" / "baselines"
    baselines.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = baselines / "redirect"
    with (
        _directory_redirect(link, outside),
        pytest.raises(ValueError, match="incomplete report cannot replace a canonical baseline"),
    ):
        benchmark.write_benchmark_report(report, link / "failure.json", baselines)


def test_cli_sanitizes_report_output_failures(monkeypatch, capsys) -> None:
    from scripts import run_performance_baseline as command

    monkeypatch.setattr(
        command,
        "_load_settings",
        lambda: SimpleNamespace(
            benchmark_database_url="postgresql://benchmark", database_url="postgresql://application"
        ),
    )
    monkeypatch.setattr(command, "run_performance_baseline", lambda config: object())
    monkeypatch.setattr(
        command,
        "write_benchmark_report",
        lambda *args: (_ for _ in ()).throw(ValueError("/private/benchmark-result")),
    )

    assert command.main(["--allow-reset"]) == 1
    assert capsys.readouterr().out == "benchmark incomplete: output_write: ValueError\n"


def test_cli_sanitizes_settings_failure_and_never_exposes_secret(monkeypatch, capsys, tmp_path) -> None:
    from scripts import run_performance_baseline as command

    monkeypatch.setattr(command, "_load_settings", lambda: (_ for _ in ()).throw(RuntimeError("postgresql://secret@host")))
    assert command.main(["--diagnostic-output", str(tmp_path / "diagnostic.json")]) == 1
    rendered = capsys.readouterr().out
    assert rendered == "benchmark incomplete: target_validation: RuntimeError\n"
    assert "secret" not in rendered
