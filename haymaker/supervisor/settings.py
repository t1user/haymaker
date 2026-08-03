"""Connection and recovery settings for the supervisor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import ib_insync as ibi


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
        log_datafarm_status: Whether to log non-actionable data-farm status messages.
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
    log_datafarm_status: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Construct connection settings from a plain configuration section.

        Args:
            values: Merged ``connection`` configuration section.

        Returns:
            Settings with a reconstructed IB probe contract.
        """

        options = dict(values)
        probe_contract = options.get("probe_contract")
        if isinstance(probe_contract, Mapping):
            options["probe_contract"] = ibi.Contract.create(**dict(probe_contract))
        return cls(**options)
