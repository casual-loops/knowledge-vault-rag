"""Production vault privacy preflight validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_rag.config import settings
from knowledge_rag.ingest.markdown import (
    discover_markdown_files,
    load_markdown,
)
from knowledge_rag.ingestion_config import IngestionConfig
from knowledge_rag.logging_utils import safe_note_reference
from knowledge_rag.policy import (
    AIAccess,
    evaluate_note_policy,
    note_type_is_excluded,
    path_is_excluded,
)


@dataclass(frozen=True, slots=True)
class PreflightFailure:
    """One sanitized production preflight failure."""

    scope: str
    reference: str
    error_type: str


@dataclass(frozen=True, slots=True)
class ProductionPreflightSummary:
    """Aggregate privacy validation result."""

    discovered: int
    excluded_by_path: int
    excluded_by_type: int
    explicit_allowed: int
    explicit_local_only: int
    explicit_excluded: int
    conservative_fallback: int
    failures: tuple[PreflightFailure, ...]

    @property
    def passed(self) -> bool:
        """Whether production indexing may proceed."""

        return not self.failures


def run_production_preflight(
    config: IngestionConfig,
) -> ProductionPreflightSummary:
    """Validate production configuration and privacy behavior."""

    failures: list[PreflightFailure] = []

    failures.extend(
        _validate_configuration(
            config,
        )
    )

    if failures:
        return ProductionPreflightSummary(
            discovered=0,
            excluded_by_path=0,
            excluded_by_type=0,
            explicit_allowed=0,
            explicit_local_only=0,
            explicit_excluded=0,
            conservative_fallback=0,
            failures=tuple(failures),
        )

    note_paths = discover_markdown_files(
        config.vault_path,
    )

    excluded_by_path = 0
    excluded_by_type = 0
    explicit_allowed = 0
    explicit_local_only = 0
    explicit_excluded = 0
    conservative_fallback = 0

    for note_path in note_paths:
        try:
            reference = safe_note_reference(
                config.vault_path,
                note_path,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                PreflightFailure(
                    scope="note",
                    reference=note_path.name or "<invalid-note-path>",
                    error_type=type(exc).__name__,
                )
            )
            continue

        try:
            if path_is_excluded(
                config.vault_path,
                note_path,
                settings.excluded_paths,
            ):
                excluded_by_path += 1
                continue

            document = load_markdown(
                note_path,
            )

            if note_type_is_excluded(
                document.metadata,
                settings.excluded_note_types,
            ):
                excluded_by_type += 1
                continue

            raw_access = document.metadata.get(
                "ai_access",
            )

            policy = evaluate_note_policy(
                document.metadata,
            )

            if isinstance(raw_access, str) and _is_valid_ai_access(raw_access):
                access = AIAccess(
                    raw_access.strip().lower()
                )

                if access is AIAccess.ALLOWED:
                    explicit_allowed += 1

                    if not policy.may_use_external_embeddings:
                        failures.append(
                            PreflightFailure(
                                scope="policy",
                                reference=reference,
                                error_type="AllowedPolicyMismatch",
                            )
                        )

                elif access is AIAccess.LOCAL_ONLY:
                    explicit_local_only += 1

                    if (
                        policy.may_use_external_embeddings
                        or policy.may_send_to_external_generation
                    ):
                        failures.append(
                            PreflightFailure(
                                scope="policy",
                                reference=reference,
                                error_type="LocalOnlyExternalEligibility",
                            )
                        )

                else:
                    explicit_excluded += 1

                    if policy.may_index:
                        failures.append(
                            PreflightFailure(
                                scope="policy",
                                reference=reference,
                                error_type="ExcludedPolicyMismatch",
                            )
                        )

            else:
                conservative_fallback += 1

                if (
                    policy.ai_access is not AIAccess.LOCAL_ONLY
                    or policy.may_use_external_embeddings
                    or policy.may_send_to_external_generation
                ):
                    failures.append(
                        PreflightFailure(
                            scope="policy",
                            reference=reference,
                            error_type="UnsafePolicyFallback",
                        )
                    )

        except Exception as exc:  # noqa: BLE001
            failures.append(
                PreflightFailure(
                    scope="note",
                    reference=reference,
                    error_type=type(exc).__name__,
                )
            )

    return ProductionPreflightSummary(
        discovered=len(note_paths),
        excluded_by_path=excluded_by_path,
        excluded_by_type=excluded_by_type,
        explicit_allowed=explicit_allowed,
        explicit_local_only=explicit_local_only,
        explicit_excluded=explicit_excluded,
        conservative_fallback=conservative_fallback,
        failures=tuple(failures),
    )


def _validate_configuration(
    config: IngestionConfig,
) -> list[PreflightFailure]:
    """Return failures for unsafe production configuration."""

    failures: list[PreflightFailure] = []

    if not config.production:
        failures.append(
            PreflightFailure(
                scope="configuration",
                reference="production",
                error_type="ProductionModeRequired",
            )
        )

    if not config.production_opt_in:
        failures.append(
            PreflightFailure(
                scope="configuration",
                reference="production",
                error_type="ProductionOptInRequired",
            )
        )

    if not config.vault_path.is_absolute():
        failures.append(
            PreflightFailure(
                scope="configuration",
                reference="vault_path",
                error_type="AbsoluteVaultPathRequired",
            )
        )

    if (
        not config.vault_path.exists()
        or not config.vault_path.is_dir()
    ):
        failures.append(
            PreflightFailure(
                scope="configuration",
                reference="vault_path",
                error_type="InvalidVaultPath",
            )
        )

    for excluded in settings.excluded_paths:
        if not _safe_relative_exclusion(
            excluded,
        ):
            failures.append(
                PreflightFailure(
                    scope="configuration",
                    reference="excluded_paths",
                    error_type="UnsafePathExclusion",
                )
            )
            break

    for note_type in settings.excluded_note_types:
        if (
            not isinstance(note_type, str)
            or not note_type.strip()
        ):
            failures.append(
                PreflightFailure(
                    scope="configuration",
                    reference="excluded_note_types",
                    error_type="InvalidNoteTypeExclusion",
                )
            )
            break

    return failures


def _safe_relative_exclusion(
    value: Any,
) -> bool:
    """Whether an exclusion is a non-empty vault-relative path."""

    if not isinstance(value, str):
        return False

    normalized = value.strip()

    if not normalized:
        return False

    path = Path(normalized)

    if path.is_absolute():
        return False

    return ".." not in path.parts


def _is_valid_ai_access(
    value: Any,
) -> bool:
    """Whether raw frontmatter explicitly declares a valid policy."""

    if not isinstance(value, str):
        return False

    try:
        AIAccess(
            value.strip().lower()
        )
    except ValueError:
        return False

    return True