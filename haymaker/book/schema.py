"""Version gate for framework-owned Book documents."""

from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 1


def require_schema(document: Mapping[str, Any]) -> None:
    """Reject unversioned/incompatible accounting before any recovery writes."""
    version = document.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(
            f"Book schema version {version!r} is incompatible; standalone conversion required"
        )
