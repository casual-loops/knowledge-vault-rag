import json
from collections.abc import Mapping
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

DOCUMENT_NAMESPACE = UUID("3de7f156-0d87-4b4c-9ab6-3fb7838d8f49")


def document_identity(vault_path: Path, note_path: Path) -> str:
    relative_path = note_path.resolve().relative_to(vault_path.resolve())
    normalized = relative_path.as_posix().lower()
    return str(uuid5(DOCUMENT_NAMESPACE, normalized))


def content_hash(
    content: str,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable digest for persisted note content and metadata."""
    payload = json.dumps(
        {"content": content, "metadata": dict(metadata or {})},
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")
