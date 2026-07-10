"""Connection supervisor implementation package."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from haymaker.config import CONFIG

from .codes import (
    BROKER_CONNECTIVITY_LOST_CODES,
    DATA_LOST_CODE,
    DATA_MAINTAINED_CODE,
    LIVE_UPDATE_FAILURE_CODE,
    SOCKET_RESET_CODE,
    SUPERVISOR_OWNED_BROKER_CODES,
    WEAK_DATA_FARM_CODES,
)
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


if TYPE_CHECKING:
    from .supervisor import ConnectionSupervisor
elif _configured_supervisor_mode(CONFIG) == "onion":
    from .supervisor_one import ConnectionSupervisor
else:
    from .supervisor import ConnectionSupervisor

__all__ = [
    "BROKER_CONNECTIVITY_LOST_CODES",
    "ConnectionSettings",
    "ConnectionSupervisor",
    "DATA_LOST_CODE",
    "DATA_MAINTAINED_CODE",
    "LIVE_UPDATE_FAILURE_CODE",
    "SOCKET_RESET_CODE",
    "SupervisorMode",
    "SupervisorWorkload",
    "SUPERVISOR_OWNED_BROKER_CODES",
    "WEAK_DATA_FARM_CODES",
]
