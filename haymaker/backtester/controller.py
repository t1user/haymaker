"""Backtest-specific adapter around Haymaker's live Controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import ib_insync as ibi

from haymaker.components.messages import PositionTarget
from haymaker.controller import Controller

from .broker import SimulatedIB


@dataclass(eq=False)
class SimulationController(Controller):
    """Reuse Controller order and fill paths without live lifecycle policy.

    Market availability is defined by replay data, not by persisted exchange
    schedules. A new order is accepted only when its concrete Contract has an
    actual bar at the current clock point. An already-pending order remains
    eligible for that Contract's next actual bar; consequently live delayed
    target verification is intentionally disabled during replay.
    """

    def __post_init__(self) -> None:
        """Wire normal Controller callbacks and enable them for replay."""

        super().__post_init__()
        self.release_hold()

    async def run(self) -> bool:
        """Enable replay callbacks without reconciliation or runtime timers.

        Returns:
            Always ``True`` after releasing Controller's startup hold.
        """

        self.release_hold()
        return True

    async def onData(
        self,
        target: PositionTarget,
        execution_model_name: str | None = None,
    ) -> None:
        """Validate accepted-target envelopes without live delayed checks.

        Args:
            target: Target accepted by an execution model.
            execution_model_name: Stable model identity supplied by the graph.

        Raises:
            TypeError: If ``target`` is not a PositionTarget.
            ValueError: If the execution model identity is absent.
        """

        if not isinstance(target, PositionTarget):
            raise TypeError("Controller accepts only PositionTarget")
        if not execution_model_name:
            raise ValueError("execution_model_name is required")

    def verify_market_open(self, contract: ibi.Contract) -> bool:
        """Accept a new order only during an observed Contract session."""

        if not isinstance(contract, ibi.Contract):
            raise TypeError("contract must be an ib_insync.Contract")
        if not contract.conId:
            raise ValueError("backtest orders require a concrete non-zero conId")
        return cast(SimulatedIB, self.ib).has_session(contract)

    def verify_position_with_broker(self, contract: ibi.Contract) -> None:
        """Skip live immediate position verification during pending replay work."""

        del contract


__all__ = ["SimulationController"]
