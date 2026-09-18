from pathlib import Path

import pytest

from knowledge_rag.logging_utils import safe_note_reference


def test_safe_note_reference_is_vault_relative() -> None:
    vault = Path("examples/sample-vault")
    note = vault / "20 Learning" / "Azure RBAC.md"

    result = safe_note_reference(vault, note)

    assert result == "20 Learning/Azure RBAC.md"


def test_safe_note_reference_does_not_include_absolute_path() -> None:
    vault = Path("examples/sample-vault")
    note = vault / "20 Learning" / "Azure RBAC.md"

    result = safe_note_reference(vault, note)

    assert str(vault.resolve()) not in result


def test_safe_note_reference_sanitizes_path_outside_vault(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.md"

    with pytest.raises(ValueError) as exc_info:
        safe_note_reference(vault, outside)

    message = str(exc_info.value)
    assert str(vault) not in message
    assert str(outside) not in message
