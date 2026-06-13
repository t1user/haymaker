"""Connection supervisor implementation package."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from haymaker.config import CONFIG

from .settings import ConnectionSettings, SupervisorWorkload

SupervisorMode = Literal["state", "onion"]


def _configured_supervisor_mode(config: Mapping[str, Any]) -> SupervisorMode:
    """Return the configured supervisor implementation name."""

    app_config = config.get("app") or {}
    if not isinstance(app_config, Mapping):
        app_config = {}
    mode = app_config.get("supervisor", config.get("supervisor", "state"))
    if mode not in ("state", "onion"):
        raise ValueError(
            "Unknown supervisor implementation "
            f"{mode!r}; expected 'state' or 'onion'."
        )
    return mode


if _configured_supervisor_mode(CONFIG) == "onion":
    from .supervisor_one import ConnectionSupervisor
else:
    from .supervisor import ConnectionSupervisor

__all__ = [
    "ConnectionSettings",
    "ConnectionSupervisor",
    "SupervisorMode",
    "SupervisorWorkload",
]
