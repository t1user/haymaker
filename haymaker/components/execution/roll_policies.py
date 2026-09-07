"""Choose when and where to roll; executors own broker work and recovery."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import ib_insync as ibi

from ...contract_selector import FutureSelector
from ...validators import non_empty_string, qualified_contract
from .future_roll import RollHolding


@dataclass(frozen=True, kw_only=True)
class RollDecision:
    """Request relocation of a complete holding within its registered chain.

    Args:
        destination: Qualified Future from the holding's registered blueprint.
        occurrence: Optional stable schedule label, such as ``2026-09-early``.
            Use it for fixed schedules whose condition remains true after a
            roll. Book remembers completion for the source (one-to-one) or
            old and replacement Contracts (direct), preventing repeated rolls
            for that occurrence. Use a new label for the next scheduled roll.
            With None, the policy's condition alone must prevent repetition.

    This is not a partial allocation or a general spread-trading instruction.
    """

    destination: ibi.Future
    occurrence: str | None = None

    def __post_init__(self) -> None:
        """Validate the selected endpoint and optional schedule identity."""
        if not isinstance(self.destination, ibi.Future):
            raise TypeError("destination must be a Future")
        qualified_contract(self.destination, "destination")
        if self.occurrence is not None:
            non_empty_string(self.occurrence, "occurrence")


class FutureRollPolicy(ABC):
    """Define a roll trigger and destination without submitting broker orders.

    Override :meth:`plan` and inject an instance as an execution model's
    ``roll_policy``. Default models use :class:`PastToActiveRollPolicy`.
    Policies run synchronously at Controller checks, not on each price update.
    Use selector wrappers' ``roll_day`` or expiry information for an early
    trigger, or your own calendar for a fixed schedule. Return None when not
    due; prefer due-or-overdue checks so a disconnected interval is not missed.
    """

    @abstractmethod
    def plan(
        self, holding: RollHolding, selector: FutureSelector, *, now: datetime
    ) -> RollDecision | None:
        """Return a due roll using a selector evaluated at this check's time.

        Args:
            holding: Fill-accounted quantity and ownership of a concrete Future.
            selector: Full chain, refreshed for this check without mutating the
                graph's selector. Its broker dates are timezone-naive UTC.
            now: Timezone-aware UTC check time. Normalize calendar comparisons
                explicitly when combining it with selector dates.

        Returns:
            A destination in the same registered chain, or None. Do not perform
            I/O or mutate Book here; accepted endpoints are persisted by the
            coordinator and recovery does not recalculate them.
        """


class PastToActiveRollPolicy(FutureRollPolicy):
    """Roll only selector ``past_contracts`` into ACTIVE.

    NEXT and every later still-eligible expiry are retained. Non-futures never
    enter this policy. The replacement is not past, so no occurrence label is
    needed to prevent a repeated roll.
    """

    def plan(
        self, holding: RollHolding, selector: FutureSelector, *, now: datetime
    ) -> RollDecision | None:
        """Select ACTIVE only after the holding's selector roll date."""
        if any(
            wrapper.contract.conId == holding.contract.conId
            for wrapper in selector.past_contracts
        ):
            return RollDecision(destination=selector.active_contract)
        return None


__all__ = ["FutureRollPolicy", "PastToActiveRollPolicy", "RollDecision"]
