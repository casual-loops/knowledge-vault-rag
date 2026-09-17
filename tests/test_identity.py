from datetime import date
from pathlib import Path

from knowledge_rag.ingest.identity import content_hash, document_identity


def test_document_identity_is_stable() -> None:
    vault = Path("examples/sample-vault")
    note = vault / "20 Learning" / "Azure RBAC.md"

    first = document_identity(vault, note)
    second = document_identity(vault, note)

    assert first == second


def test_document_identity_changes_with_path() -> None:
    vault = Path("examples/sample-vault")

    first = document_identity(
        vault,
        vault / "20 Learning" / "Azure RBAC.md",
    )
    second = document_identity(
        vault,
        vault / "40 Reference" / "Azure RBAC.md",
    )

    assert first != second


def test_content_hash_is_stable() -> None:
    assert content_hash("hello") == content_hash("hello")


def test_content_hash_changes_with_content() -> None:
    assert content_hash("hello") != content_hash("hello world")


def test_content_hash_changes_with_metadata() -> None:
    content = "# Identity\n\nUnchanged note body."

    first = content_hash(
        content,
        {"ai_access": "local-only", "topic": ["identity"]},
    )
    second = content_hash(
        content,
        {"ai_access": "allowed", "topic": ["identity"]},
    )

    assert first != second


def test_content_hash_ignores_top_level_and_nested_metadata_key_order() -> None:
    content = "# Identity\n\nUnchanged note body."

    first = content_hash(
        content,
        {
            "topic": ["identity"],
            "policy": {"ai_access": "allowed", "status": "active"},
            "reviewed": date(2026, 9, 17),
        },
    )
    second = content_hash(
        content,
        {
            "reviewed": date(2026, 9, 17),
            "policy": {"status": "active", "ai_access": "allowed"},
            "topic": ["identity"],
        },
    )

    assert first == second
