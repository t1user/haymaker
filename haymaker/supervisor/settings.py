"""Shared supervisor settings and workload protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import ib_insync as ibi


class SupervisorWorkload(Protocol):
    """Workload run by a connection supervisor after IB is usable."""

    async def start(self) -> None:
        """Start or resume work after a usable IB connection is available."""

    async def stop(self, reason: str) -> None:
        """Release active work before the supervisor reconnects or exits."""


@dataclass(frozen=True)
class ConnectionSettings:
    """Connection and recovery settings for a connection supervisor.

    Attributes:
        host: Hostname or IP address of the TWS/IB Gateway API endpoint.
        port: Port number of the TWS/IB Gateway API endpoint.
        client_id: IB API client ID used for this supervised socket.
        connect_timeout: Seconds to wait for one socket connection attempt.
        retry_delay: Seconds to wait between failed connection attempts.
        app_timeout: Seconds of no IB traffic before running connection health checks.
        probe_contract: Contract used for the small historical-data readiness probe.
        probe_timeout: Seconds to wait for the readiness probe to complete.
        connection_lost_retry_delay: Seconds to wait after lost connection before reconnecting.
        auto_recovery_grace_period: Seconds to wait for broker-side recovery before reconnecting.
        restart_on_recovered_connection: Whether to restart even after IB reports data was maintained.
    """

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 0
    connect_timeout: float = 15
    retry_delay: float = 30
    app_timeout: float = 90
    probe_contract: ibi.Contract = field(default_factory=lambda: ibi.Forex("EURUSD"))
    probe_timeout: float = 15
    connection_lost_retry_delay: float = 90
    auto_recovery_grace_period: float = 120
    restart_on_recovered_connection: bool = False

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], client_id: int
    ) -> ConnectionSettings:
        """Create connection settings from a flat config mapping.

        Args:
            config: Mapping with connection keys directly available.
            client_id: IB API client ID chosen by the caller.
        """

        return cls(
            host=config.get("host", "127.0.0.1"),
            port=config.get("port", 4002),
            client_id=client_id,
            connect_timeout=config.get("connectTimeout", 15),
            retry_delay=config.get("retryDelay", 30),
            app_timeout=config.get("appTimeout", 90),
            probe_contract=config.get("probeContract") or ibi.Forex("EURUSD"),
            probe_timeout=config.get("probeTimeout", 15),
            connection_lost_retry_delay=config.get("connection_lost_retry", 90),
            auto_recovery_grace_period=config.get("auto_recovery_grace_period", 120),
            restart_on_recovered_connection=config.get(
                "restart_on_recovered_connection", False
            ),
        )
