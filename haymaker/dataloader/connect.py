"""Dataloader connection lifecycle management."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger
from typing import Literal, TypeAlias

from ib_insync import IB, util

from haymaker.config import CONFIG
from haymaker.supervisor import ConnectionSupervisor, SupervisorState

log = getLogger(__name__)

Mode: TypeAlias = Literal["reconnect", "wait"]


@dataclass
class DataloaderConnection:
    """Run dataloader work under the shared IB connection supervisor.

    Args:
        ib: Interactive Brokers client used by the dataloader.
        func: Async dataloader workload to run after connection.
        cleanup: Optional callback used to release work after disconnection.
        run_mode: Re-run work after reconnect, or wait for in-place recovery.
    """

    ib: IB
    func: Callable
    cleanup: Callable | None = None
    run_mode: Mode = "reconnect"
    supervisor: ConnectionSupervisor = field(init=False)
    _started: bool = field(default=False, init=False)
    _work_task: asyncio.Task | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        client_id = CONFIG.get("clientId") or random.randint(60, 90)
        self.supervisor = ConnectionSupervisor(
            ib=self.ib,
            on_connected=self.on_connected,
            on_restarting=self.on_restarting,
            host=CONFIG.get("host", "localhost"),
            port=CONFIG.get("port", 4002),
            client_id=client_id,
            connect_timeout=CONFIG.get("connectTimeout", 2),
            restart_delay=CONFIG.get("restart_time", 60),
            retry_delay=CONFIG.get("retryDelay", 60),
            app_timeout=0,
            probe_on_connect=False,
            auto_recovery_grace_period=CONFIG.get("auto_recovery_grace_period", 120),
            recovery_warning_after=CONFIG.get("recovery_warning_after", 300),
            recovery_warning_interval=CONFIG.get("recovery_warning_interval", 900),
        )

    async def on_connected(self) -> None:
        """Start or resume dataloader work after connection."""

        if self.run_mode == "wait" and self._started:
            log.debug("Reconnected. Waiting for IB to resume sending data.")
            return

        if self._work_task and not self._work_task.done():
            log.debug("Dataloader work is still active after reconnection.")
            return

        self._started = True
        log.debug(f"Running {self.func}.")
        self._work_task = asyncio.create_task(self._run_work(), name="dataloader-work")

    async def _run_work(self) -> None:
        """Run one dataloader workload and stop after normal completion."""

        try:
            await self.func()
        except (asyncio.CancelledError, ConnectionError):
            log.debug("Dataloader work interrupted during connection recovery.")
        except Exception as exc:
            log.exception(f"Dataloader work failed: {exc}")
        finally:
            if self.supervisor.state != SupervisorState.RESTARTING:
                self.supervisor.stop()

    def on_restarting(self, reason: str) -> None:
        """Release current work before reconnecting when configured to rerun."""

        log.debug(f"Restarting dataloader connection: {reason}")
        if self.run_mode == "reconnect" and self.cleanup:
            self.cleanup()

    def run(self) -> None:
        """Run until the dataloader workload finishes."""

        util.run(self.supervisor.run())


def connection(
    ib: IB,
    func: Callable,
    cleanup: Callable | None = None,
    run_mode: Mode = "reconnect",
) -> DataloaderConnection:
    """Run dataloader work with shared connection supervision."""

    if run_mode not in ("reconnect", "wait"):
        raise ValueError(
            f"Unknown mode: {run_mode}. The gateway-managing watchdog mode "
            "is no longer supported."
        )

    log.debug(f"Running in {run_mode} mode.")
    dataloader_connection = DataloaderConnection(ib, func, cleanup, run_mode)
    dataloader_connection.run()
    return dataloader_connection
