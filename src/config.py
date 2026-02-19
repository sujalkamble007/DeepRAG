# src/config.py
"""
Configuration loader for DeepRAG.
Reads settings from a TOML config file and supports secrets from
environment variables or .streamlit/secrets.toml.
"""

import os
import sys

# For Python < 3.11, use the tomli back-port; otherwise use the stdlib tomllib
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        import tomllib  # type: ignore

_CONFIG_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "config.toml"),      # src/config.toml
    os.path.join(os.path.dirname(__file__), "..", "config.toml"), # repo root config.toml
]

_SECRETS_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml"),
]


class _Config:
    """Thin wrapper around a nested dict loaded from TOML."""

    def __init__(self, data: dict, secrets: dict):
        self._data = data
        self._secrets = secrets

    # ── dot-path access (e.g. "model.embedding_model") ──────────
    def get(self, dotted_key: str, default=None):
        """Return a value from the config by dot-separated key path."""
        keys = dotted_key.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    # ── secret access ────────────────────────────────────────────
    def get_secret(self, name: str) -> str:
        """Return a secret value, checking env-vars first, then secrets.toml."""
        # 1. Environment variable (upper-case)
        env_val = os.environ.get(name.upper()) or os.environ.get(name)
        if env_val:
            return env_val

        # 2. Streamlit secrets.toml
        if name in self._secrets:
            return str(self._secrets[name])

        raise ValueError(
            f"Secret '{name}' not found. Set the {name.upper()} environment "
            f"variable or add it to .streamlit/secrets.toml"
        )


def _load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config() -> _Config:
    """Locate and parse the TOML config + secrets files."""
    # ── main config ──────────────────────────────────────────────
    data: dict = {}
    for p in _CONFIG_SEARCH_PATHS:
        resolved = os.path.abspath(p)
        if os.path.isfile(resolved):
            data = _load_toml(resolved)
            break

    # ── secrets ──────────────────────────────────────────────────
    secrets: dict = {}
    for p in _SECRETS_SEARCH_PATHS:
        resolved = os.path.abspath(p)
        if os.path.isfile(resolved):
            secrets = _load_toml(resolved)
            break

    return _Config(data, secrets)
