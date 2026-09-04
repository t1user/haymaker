"""Controller-owned futures-roll discovery and recovery coordination."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import eventkit as ev  # type: ignore
import ib_insync as ibi

from haymaker.book import (
    FutureRollMode,
    FutureRollStage,
    RollParticipant,
    RollState,
)
from haymaker.components.execution.future_roll import (
    BracketFutureRollExecutor,
    DirectFutureRollExecutor,
    FutureRollExecutor,
    RollHolding,
)

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


class FutureRoller:
    """Discover stale futures holdings and delegate durable execution.

    The Controller owns one instance for the entire process. Target execution
    models register exactly one direct or bracket executor family during
    strategy construction. Daily callbacks discover new work; recovery always
    resumes persisted work before creating another plan for the series.
    """

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.book = controller.book
        self.future_roll_policies: dict[str, bool] = {}
        self.executor: FutureRollExecutor | None = None
        self.completedEvent = ev.Event("futureRollCompleted")

    def register_executor(
        self,
        mode: FutureRollMode,
        executor: FutureRollExecutor | None = None,
    ) -> FutureRollExecutor:
        """Register or reuse the process's one mode-specific executor.

        Args:
            mode: Mutually exclusive target-accounting mode.
            executor: Optional custom executor. When omitted, the built-in
                implementation for the selected mode is created.

        Raises:
            ValueError: If another mode or a different instance was already
                registered.
        """

        if not isinstance(mode, FutureRollMode):
            raise TypeError("mode must be a FutureRollMode")
        if self.executor is not None:
            if self.executor.mode is not mode:
                raise ValueError(
                    "Direct and one-to-one futures-roll modes cannot be mixed"
                )
            if executor is not None and executor is not self.executor:
                raise ValueError("A different FutureRollExecutor is already registered")
            return self.executor
        selected = executor or (
            DirectFutureRollExecutor()
            if mode is FutureRollMode.DIRECT
            else BracketFutureRollExecutor()
        )
        if not isinstance(selected, FutureRollExecutor):
            raise TypeError("executor must be a FutureRollExecutor")
        if selected.mode is not mode:
            raise ValueError(
                f"{type(selected).__name__} does not support {mode.value} mode"
            )
        selected.bind(self.controller)
        selected.completedEvent += self.onRollCompletedEvent
        self.executor = selected
        return selected

    def set_policies(self, policies: Mapping[str, bool]) -> None:
        """Replace one-to-one automatic-roll declarations."""

        self.future_roll_policies = dict(policies)

    def recover(self) -> bool:
        """Resume every non-complete persisted roll.

        Returns:
            True when recoverable roll state existed.
        """

        states = self.book.roll_states(active_only=True)
        if not states:
            return False
        blocked = tuple(
            state for state in states if state.stage is FutureRollStage.BLOCKED
        )
        if blocked:
            series = ", ".join(state.series_key for state in blocked)
            raise RuntimeError(f"Blocked futures roll state requires review: {series}")
        executor = self._required_executor()
        for state in states:
            executor.recover(state)
        return True

    def roll(self, *args: object) -> None:
        """Resume existing work, then plan every currently stale series."""

        executor = self.executor
        if executor is None:
            if self.book.roll_states(active_only=True):
                self._required_executor()
            log.debug("No FutureRollExecutor is registered.")
            return
        self.recover()
        holdings = executor.holdings()
        if not holdings:
            log.debug("No futures positions require rolling.")
            return
        grouped = self._stale_holdings(holdings)
        for series_key, by_contract in grouped.items():
            current = self.book.roll_state(series_key)
            if current is not None and current.stage is not FutureRollStage.COMPLETE:
                continue
            old_contract = min(
                by_contract,
                key=lambda contract: (
                    contract.lastTradeDateOrContractMonth,
                    contract.localSymbol,
                    contract.conId,
                ),
            )
            old_holdings = by_contract[old_contract]
            new_contract = self.controller.contract_registry.active_for_series(
                series_key
            )
            if old_contract.conId == new_contract.conId:
                continue
            if not self._broker_quantity_matches(old_contract):
                self._persist_blocked_discovery(
                    series_key,
                    old_contract,
                    new_contract,
                    old_holdings,
                    "Book quantity does not match broker position",
                )
                continue
            details = self.controller.contract_registry.get_details(new_contract)
            if details is not None and not details.is_open():
                log.warning(
                    "Replacement Future is closed; roll deferred for %s",
                    new_contract.localSymbol,
                )
                continue
            try:
                state = executor.create_state(
                    series_key,
                    old_contract,
                    new_contract,
                    old_holdings,
                )
            except (TypeError, ValueError) as exc:
                self._persist_blocked_discovery(
                    series_key,
                    old_contract,
                    new_contract,
                    old_holdings,
                    str(exc),
                )
                continue
            self.book.update_roll(state)
            log.warning(
                "Rolling futures series %s from %s to %s",
                series_key,
                old_contract.localSymbol,
                new_contract.localSymbol,
            )
            executor.advance(state)

    def _stale_holdings(
        self, holdings: Sequence[RollHolding]
    ) -> dict[str, dict[ibi.Future, list[RollHolding]]]:
        grouped: dict[str, dict[ibi.Future, list[RollHolding]]] = defaultdict(
            lambda: defaultdict(list)
        )
        direct_keys: dict[str, set[str]] = defaultdict(set)
        for holding in holdings:
            try:
                series_key = self.controller.contract_registry.series_key(
                    holding.contract
                )
                current = self.controller.contract_registry.current_for_series(
                    series_key
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.critical(
                    "Cannot identify futures series for conId=%s: %s",
                    holding.contract.conId,
                    exc,
                )
                continue
            if holding.target_key is not None:
                direct_keys[series_key].add(holding.target_key)
            if all(holding.contract.conId != contract.conId for contract in current):
                grouped[series_key][holding.contract].append(holding)
        for series_key, keys in direct_keys.items():
            if len(keys) > 1 and series_key in grouped:
                by_contract = grouped.pop(series_key)
                old_contract = next(iter(by_contract))
                new_contract = self.controller.contract_registry.active_for_series(
                    series_key
                )
                holdings_for_series = tuple(
                    holding for values in by_contract.values() for holding in values
                )
                self._persist_blocked_discovery(
                    series_key,
                    old_contract,
                    new_contract,
                    holdings_for_series,
                    "Direct series has multiple live target_key values",
                )
        return grouped

    def _broker_quantity_matches(self, contract: ibi.Future) -> bool:
        return self.book.aggregate_quantity(
            contract
        ) == self.controller.trader.position_for_contract(contract)

    def _persist_blocked_discovery(
        self,
        series_key: str,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
        holdings: Sequence[RollHolding],
        reason: str,
    ) -> None:
        executor = self._required_executor()
        participants = tuple(
            RollParticipant(
                execution_model_name=holding.execution_model_name,
                quantity=holding.quantity,
                source_key=holding.source_key,
                target_key=holding.target_key,
                position_id=holding.position_id,
            )
            for holding in holdings
        )
        state = RollState(
            series_key=series_key,
            mode=executor.mode,
            executor_name=executor.name,
            old_contract=old_contract,
            new_contract=new_contract,
            participants=participants,
            stage=FutureRollStage.BLOCKED,
            failure_reason=reason,
            updated_at=datetime.now(timezone.utc),
        )
        self.book.update_roll(state)
        log.critical("Futures roll blocked for series %s: %s", series_key, reason)

    def _required_executor(self) -> FutureRollExecutor:
        if self.executor is None:
            raise RuntimeError(
                "Persisted or eligible futures roll has no registered executor"
            )
        return self.executor

    def onRollCompletedEvent(self, state: RollState) -> None:
        """Notify target models and discover the next stale expiry."""

        self.completedEvent.emit(state)
        self.roll()

    def __repr__(self) -> str:
        return (
            f"FutureRoller(executor={self.executor!r}, "
            f"future_roll_policies={self.future_roll_policies!r})"
        )
