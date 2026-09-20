import json
from dataclasses import replace
from pathlib import Path

import pytest

from knowledge_rag.benchmark_corpus import (
    generate_synthetic_vault,
    load_benchmark_workload,
    modify_synthetic_vault,
)
from knowledge_rag.ingest.chunker import chunk_markdown
from knowledge_rag.ingest.markdown import load_markdown

WORKLOAD_PATH = Path("benchmarks/workload.json")


def test_workload_has_expected_tiers_queries_and_modes() -> None:
    workload = load_benchmark_workload(WORKLOAD_PATH)

    assert workload.tiers == (100, 1000, 10000)
    assert len(workload.queries) == 9
    assert {query.retrieval_mode for query in workload.queries} == {
        "semantic",
        "lexical",
        "hybrid",
    }
    assert all(
        sum(query.retrieval_mode == mode for query in workload.queries) == 3
        for mode in ("semantic", "lexical", "hybrid")
    )


def test_generation_is_byte_identical_and_manifest_paths_are_relative(tmp_path: Path) -> None:
    workload = load_benchmark_workload(WORKLOAD_PATH)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = generate_synthetic_vault(first_root, workload, 100)
    second = generate_synthetic_vault(second_root, workload, 100)

    assert first == second
    assert [
        path.relative_to(first_root).as_posix()
        for path in sorted(first_root.rglob("*.md"))
    ] == list(first.paths)
    for relative_path in first.paths:
        assert (first_root / relative_path).read_bytes() == (
            second_root / relative_path
        ).read_bytes()
        assert not Path(relative_path).is_absolute()


def test_100_note_generation_has_three_hundred_chunks(tmp_path: Path) -> None:
    workload = load_benchmark_workload(WORKLOAD_PATH)
    root = tmp_path / "vault"
    manifest = generate_synthetic_vault(root, workload, 100)

    assert len(list(root.rglob("*.md"))) == 100
    assert manifest.chunk_count == 300
    assert all(
        len(chunk_markdown(load_markdown(root / path).content)) == 3
        for path in manifest.paths
    )


def test_mutation_changes_exactly_ten_files_and_preserves_chunks(tmp_path: Path) -> None:
    workload = load_benchmark_workload(WORKLOAD_PATH)
    root = tmp_path / "vault"
    manifest = generate_synthetic_vault(root, workload, 100)

    changed = modify_synthetic_vault(root, manifest, 0.1)

    assert len(changed) == 10
    assert changed == tuple(manifest.paths[index] for index in range(0, 100, 10))
    assert all(
        len(chunk_markdown(load_markdown(root / path).content)) == 3
        for path in changed
    )


def test_mutation_uses_sorted_path_position_not_manifest_order(tmp_path: Path) -> None:
    workload = load_benchmark_workload(WORKLOAD_PATH)
    root = tmp_path / "vault"
    manifest = generate_synthetic_vault(root, workload, 100)
    shuffled = replace(manifest, paths=tuple(reversed(manifest.paths)))

    changed = modify_synthetic_vault(root, shuffled, 0.1)

    assert changed == tuple(sorted(manifest.paths)[index] for index in range(0, 100, 10))


def test_mutation_rejects_unsafe_manifest_paths_without_touching_outside_root(
    tmp_path: Path,
) -> None:
    workload = load_benchmark_workload(WORKLOAD_PATH)
    root = tmp_path / "vault"
    manifest = generate_synthetic_vault(root, workload, 100)
    outside = tmp_path / "outside.md"
    outside.write_text("do not change", encoding="utf-8")

    for unsafe in ("../outside.md", "/tmp/outside.md", "synthetic\\escape.md", ""):
        unsafe_manifest = replace(manifest, paths=(unsafe, *manifest.paths[1:]))
        with pytest.raises(ValueError, match="manifest path"):
            modify_synthetic_vault(root, unsafe_manifest, 0.1)
        assert outside.read_text(encoding="utf-8") == "do not change"


def test_mutation_preflights_unselected_files_before_writing(tmp_path: Path) -> None:
    workload = load_benchmark_workload(WORKLOAD_PATH)
    root = tmp_path / "vault"
    manifest = generate_synthetic_vault(root, workload, 100)
    selected = tuple(sorted(manifest.paths)[index] for index in range(0, 100, 10))
    before = {
        path: (root / path).read_bytes()
        for path in selected
    }
    (root / manifest.paths[1]).unlink()

    with pytest.raises(ValueError, match="manifest path"):
        modify_synthetic_vault(root, manifest, 0.1)

    assert {path: (root / path).read_bytes() for path in selected} == before


def test_loader_rejects_non_ten_percent_changed_fraction(tmp_path: Path) -> None:
    payload = json.loads(WORKLOAD_PATH.read_text(encoding="utf-8"))
    payload["changed_fraction"] = 0.2
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="changed_fraction"):
        load_benchmark_workload(path)


def test_generation_fails_closed_on_nonempty_root(tmp_path: Path) -> None:
    workload = load_benchmark_workload(WORKLOAD_PATH)
    root = tmp_path / "vault"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="nonempty"):
        generate_synthetic_vault(root, workload, 100)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_generated_content_is_public_safe(tmp_path: Path) -> None:
    workload = load_benchmark_workload(WORKLOAD_PATH)
    root = tmp_path / "vault"
    generate_synthetic_vault(root, workload, 100)
    content = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.md"))

    assert "private" not in content.lower()
    assert "localhost" not in content.lower()
    assert "password" not in content.lower()
    assert "credential" not in content.lower()
    assert "api_key" not in content.lower()
    assert "postgresql://" not in content.lower()
    assert "@" not in content


def test_loader_rejects_unknown_workload_keys(tmp_path: Path) -> None:
    payload = json.loads(WORKLOAD_PATH.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="workload"):
        load_benchmark_workload(path)
