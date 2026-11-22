"""Configuration loading: non-secret settings from YAML, secrets from environment / .env.

Rules
-----
* Secrets (API keys) live only in environment variables or the git-ignored .env.
  They are never stored on the settings object in printable form.
* YAML values may reference environment variables as ${NAME}.
* configs/config.yaml (git-ignored) overrides configs/config.example.yaml.
* Placeholder values such as <YOUR_PROJECT_ID> are rejected with a clear message
  instead of producing confusing Google Cloud errors later.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PLACEHOLDER = re.compile(r"^<.*>$")


class ConfigError(ValueError):
    """Raised for missing or placeholder configuration values."""


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by walking up until ``pyproject.toml`` is found.

    Args:
        start: Directory to start the search from. Defaults to the current
            working directory.

    Returns:
        Absolute path of the directory that contains ``pyproject.toml``.

    Raises:
        ConfigError: If no ``pyproject.toml`` is found before reaching the
            filesystem root.
    """
    root = (start or Path.cwd()).resolve()
    while not (root / "pyproject.toml").exists():
        if root.parent == root:
            raise ConfigError(
                "Could not find pyproject.toml; run from inside the analytics-agent folder."
            )
        root = root.parent
    return root


def _expand(value):
    """Recursively expand ``${ENV_VAR}`` placeholders with environment values.

    Missing variables are replaced by the empty string. Non-string values
    (dicts, lists, numbers, …) are left unchanged or expanded element-wise.

    Args:
        value: A scalar, dict or list that may contain ``${NAME}`` strings.

    Returns:
        The same structure with every environment reference substituted.
    """
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _require(value: str, name: str, hint: str) -> str:
    """Return *value* only when it is non-empty and not a placeholder.

    Args:
        value: The candidate configuration value.
        name: Human-readable name used in the error message (e.g.
            ``"GOOGLE_CLOUD_PROJECT"``).
        hint: Short instruction telling the user how to set the value.

    Returns:
        The original *value* as a string.

    Raises:
        ConfigError: If *value* is empty or matches the placeholder pattern
            ``<...>``.
    """
    if not value or PLACEHOLDER.match(str(value)):
        raise ConfigError(f"{name} is not set (got {value!r}). {hint}")
    return str(value)


@dataclass(frozen=True)
class BigQuerySettings:
    """Immutable BigQuery-related configuration.

    Holds project/dataset identifiers, location, approved tables, blocked
    columns and the cost / timeout / row limits used by the query runner.
    """

    project_id: str
    location: str
    dataset_clean: str
    dataset_corrupted: str
    approved_tables: tuple
    blocked_columns: tuple
    maximum_bytes_billed: int
    timeout_seconds: int
    max_result_rows: int

    def dataset(self, variant: str = "clean") -> str:
        """Return the dataset name for the requested variant.

        Args:
            variant: Either ``"clean"`` or ``"corrupted"``. Defaults to
                ``"clean"``.

        Returns:
            The corresponding dataset id.

        Raises:
            ConfigError: If *variant* is not one of the two allowed values.
        """
        if variant not in ("clean", "corrupted"):
            raise ConfigError(
                f"variant must be 'clean' or 'corrupted', got {variant!r}"
            )
        return self.dataset_clean if variant == "clean" else self.dataset_corrupted


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"  # stable model in the Gemini API model list (verified Oct 2026)


@dataclass(frozen=True)
class GeminiSettings:
    """Non-secret Gemini settings.

    The API key itself is read from ``os.environ`` only at the moment a client
    is created (see ``analytics_agent.llm``). This object never stores the key.
    """

    model: str = DEFAULT_GEMINI_MODEL
    timeout_seconds: float = 90.0
    max_retries: int = 2
    temperature: float = 0.0
    max_output_tokens: int = 8192


@dataclass(frozen=True)
class Settings:
    """Top-level application settings assembled from YAML and the environment.

    Secrets (API keys) are never stored on this object; only presence flags
    are kept so that a safe summary can be printed.
    """

    root: Path
    bigquery: BigQuerySettings
    raw: dict = field(repr=False)
    gemini: GeminiSettings = GeminiSettings()
    # Presence flags only: the secret values are read from os.environ when needed.
    gemini_key_present: bool = False
    langfuse_enabled: bool = False

    def summary(self) -> dict:
        """Return a safe-to-print dictionary that never contains secrets.

        Returns:
            A dict with project/dataset names, limits, the Gemini model name
            and boolean flags indicating whether the API key / Langfuse are
            configured.
        """
        bq = self.bigquery
        return {
            "project_id": bq.project_id,
            "location": bq.location,
            "dataset_clean": bq.dataset_clean,
            "dataset_corrupted": bq.dataset_corrupted,
            "approved_tables": list(bq.approved_tables),
            "maximum_bytes_billed": bq.maximum_bytes_billed,
            "timeout_seconds": bq.timeout_seconds,
            "max_result_rows": bq.max_result_rows,
            "gemini_model": self.gemini.model,
            "gemini_api_key": "set" if self.gemini_key_present else "not set",
            "langfuse_enabled": self.langfuse_enabled,
        }


def load_settings(
    root: Path | None = None,
    env_file: str = ".env",
    require_project: bool = True,
) -> Settings:
    """Load and validate application settings from YAML and the environment.

    Order of precedence
    -------------------
    1. Existing process environment variables (highest).
    2. Values from the ``.env`` file (loaded with ``override=False``).
    3. Values from ``configs/config.yaml`` (or ``config.example.yaml``).
    4. Hard-coded defaults.

    Args:
        root: Project root directory. When ``None`` the root is discovered by
            searching upward for ``pyproject.toml``.
        env_file: Name of the dotenv file relative to *root* (default
            ``".env"``).
        require_project: When ``True`` (default) a missing or placeholder
            ``GOOGLE_CLOUD_PROJECT`` / ``project_id`` raises ``ConfigError``.

    Returns:
        A frozen ``Settings`` instance containing validated BigQuery and Gemini
        configuration together with presence flags for secrets.

    Raises:
        ConfigError: On missing required values, unresolved placeholders or
            inability to locate the project root.
    """
    root = root or find_project_root()
    # Existing environment variables win over .env (so CI/shell exports override the file).
    load_dotenv(root / env_file, override=False)
    cfg_path = root / "configs" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = root / "configs" / "config.example.yaml"
    raw = _expand(yaml.safe_load(cfg_path.read_text()) or {})
    bq = raw.get("bigquery", {})
    limits = bq.get("limits", {})
    datasets = bq.get("datasets", {})

    project = bq.get("project_id") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if require_project:
        project = _require(
            project,
            "GOOGLE_CLOUD_PROJECT",
            "Set it in .env (copy .env.example) or in configs/config.yaml.",
        )
    location = os.environ.get("BQ_LOCATION") or bq.get("location") or "US"

    settings = BigQuerySettings(
        project_id=project,
        location=location,
        dataset_clean=os.environ.get("BQ_DATASET")
        or datasets.get("clean", "analytics_agent_dev"),
        dataset_corrupted=os.environ.get("BQ_DATASET_CORRUPTED")
        or datasets.get("corrupted", "analytics_agent_corrupted"),
        approved_tables=tuple(bq.get("approved_tables", [])),
        blocked_columns=tuple(bq.get("blocked_columns", [])),
        maximum_bytes_billed=int(limits.get("maximum_bytes_billed", 100_000_000)),
        timeout_seconds=int(limits.get("timeout_seconds", 60)),
        max_result_rows=int(limits.get("max_result_rows", 1000)),
    )
    gem = raw.get("gemini", {}) or {}
    model = os.environ.get("GEMINI_MODEL") or gem.get("model") or DEFAULT_GEMINI_MODEL
    if PLACEHOLDER.match(model):
        model = DEFAULT_GEMINI_MODEL
    gemini = GeminiSettings(
        model=model,
        timeout_seconds=float(gem.get("timeout_seconds", 90)),
        max_retries=int(gem.get("max_retries", 2)),
        temperature=float(gem.get("temperature", 0.0)),
        max_output_tokens=int(gem.get("max_output_tokens", 8192)),
    )
    key = os.environ.get("GEMINI_API_KEY", "")
    return Settings(
        root=root,
        bigquery=settings,
        raw=raw,
        gemini=gemini,
        gemini_key_present=bool(key) and not PLACEHOLDER.match(key),
        langfuse_enabled=(
            os.environ.get("ENABLE_LANGFUSE")
            or os.environ.get("LANGFUSE_ENABLED")
            or "false"
        ).lower()
        == "true",
    )