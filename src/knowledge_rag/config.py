from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vault_path: Path = Path("examples/sample-vault")
    database_url: str = "postgresql://knowledge_rag:change-me@localhost:5432/knowledge_rag"
    benchmark_database_url: str | None = None

    excluded_paths: tuple[str, ...] = ()
    excluded_note_types: tuple[str, ...] = ()

    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    llm_model: str | None = None



settings = Settings()
