"""Application configuration.

All runtime configuration is read from environment variables (optionally via a local
``.env`` file). Secrets are held in :class:`pydantic.SecretStr` so they are never
rendered by accident in logs or Streamlit widgets.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root when running from a source checkout (editable install). In a built
#: image the package lives under ``site-packages``, so this points somewhere useless -
#: which is exactly why :func:`resolve_config_path` never trusts it on its own.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Default location of the business rule file, relative to the working directory.
DEFAULT_RULES_FILE = Path("config/business_rules.yaml")


def resolve_config_path(value: Path) -> Path:
    """Resolve a configuration file path against the places it might actually live.

    An absolute path is taken as given. A relative one is tried against the current
    working directory first (which is ``/app`` in the container and the repository root
    in development), then against :data:`PROJECT_ROOT` for the editable-install case.

    When nothing matches, the working-directory candidate is returned unchanged, so the
    error message names a path a human can recognise instead of a ``site-packages``
    directory that only exists inside the image.
    """
    if value.is_absolute():
        return value

    candidates = (Path.cwd() / value, PROJECT_ROOT / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


class Settings(BaseSettings):
    """Runtime settings for the Data Quality Copilot application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General -----------------------------------------------------------
    app_env: str = Field(default="local", description="local | docker | ci")
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False, description="Emit logs as JSON lines.")

    # --- Ingestion limits --------------------------------------------------
    max_upload_mb: float = Field(
        default=25.0, gt=0, description="Maximum accepted upload size in megabytes."
    )
    max_rows: int = Field(
        default=200_000,
        gt=0,
        description="Hard cap on rows analysed; larger files are rejected.",
    )
    max_columns: int = Field(default=500, gt=0)
    preview_rows: int = Field(default=50, gt=0)

    # --- Storage -----------------------------------------------------------
    database_url: str = Field(
        default="sqlite+pysqlite:///./dqcopilot.db",
        description=("SQLAlchemy URL. Docker Compose overrides this with the PostgreSQL service."),
    )
    persist_uploads: bool = Field(
        default=False,
        description="When False (default) the original uploaded file is never written to disk.",
    )
    persist_examples: bool = Field(
        default=False,
        description=(
            "When False (default) example cell values are stripped from findings before "
            "they are written to the database, so the audit trail records how many "
            "problems there were without recording the data itself."
        ),
    )
    persistence_enabled: bool = Field(
        default=True,
        description="Set False to run the app with no database at all (analysis still works).",
    )

    # --- Business rules ----------------------------------------------------
    rules_file: Path = Field(
        default=DEFAULT_RULES_FILE,
        description=(
            "Business rule file. Relative paths are resolved against the working "
            "directory, then against the repository root."
        ),
    )

    @field_validator("rules_file")
    @classmethod
    def _resolve_rules_file(cls, value: Path) -> Path:
        return resolve_config_path(value)

    # --- AI ----------------------------------------------------------------
    anthropic_api_key: SecretStr | None = Field(default=None)
    anthropic_model: str = Field(
        default="claude-opus-5",
        description="Override with ANTHROPIC_MODEL, e.g. claude-sonnet-5 for a cheaper run.",
    )
    ai_max_output_tokens: int = Field(
        default=4000, gt=0, description="Hard ceiling on tokens generated per AI call."
    )
    ai_timeout_seconds: float = Field(default=45.0, gt=0)
    ai_sample_rows: int = Field(
        default=5, ge=0, le=25, description="Max example values per column sent to the model."
    )
    ai_send_samples: bool = Field(
        default=True,
        description=(
            "When False, only counts and inferred types are sent to Anthropic - no cell "
            "values at all, masked or otherwise."
        ),
    )
    ai_max_columns: int = Field(default=40, gt=0)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted upload size in bytes."""
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def ai_enabled(self) -> bool:
        """True when an Anthropic API key is configured.

        When False the application still runs in *reduced mode*: every deterministic
        feature works, only the AI suggestion panel is disabled.
        """
        key = self.anthropic_api_key
        return key is not None and bool(key.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
