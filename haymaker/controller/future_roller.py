"""Controller-owned futures-roll discovery and recovery coordination."""

from __future__ import annotations

import logging
import asyncio
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from dataclasses import replace
from typing import TYPE_CHECKING

import eventkit as ev  # type: ignore
import ib_insync as ibi

from haymaker.enums import ExecutionMode
from haymaker.book import (
    FutureRollStage,
    RollParticipant,
    RollState,
)
from haymaker.components.execution.future_roll import (
    SignalFutureRollExecutor,
    ContractFutureRollExecutor,
    FutureRollExecutor,
    RollHolding,
)
from haymaker.components.execution.roll_policies import (
    FutureRollPolicy,
    PastToActiveRollPolicy,
    RollDecision,
)
from haymaker.contract_selector import FutureSelector
from haymaker.validators import aware_datetime, non_empty_string

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


class FutureRoller:
    """Discover stale futures holdings and delegate durable execution.

    The Controller owns one instance for the entire process. Target execution
    models register exactly one contract or signal executor family during
    strategy construction. Daily callbacks discover new work; recovery always
    resumes persisted work before creating another plan for the series.
    """

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.book = controller.book
        self.future_roll_policies: dict[str, bool] = {}
        self.executor: FutureRollExecutor | None = None
        self.completedEvent = ev.Event("futureRollCompleted")
        self._policies: dict[tuple[str, str], FutureRollPolicy] = {}
        self._default_policy = PastToActiveRollPolicy()
        self._checking = False

    def register_policy(
        self,
        policy: FutureRollPolicy,
        *,
        signal_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Configure one signal or contract execution model policy.

        Register during graph construction. Automatic-roll signal opt-outs
        still take precedence. Policy selection never changes accepted work.
        """
        if not isinstance(policy, FutureRollPolicy):
            raise TypeError("policy must be a FutureRollPolicy")
        if (signal_key is None) == (model_name is None):
            raise ValueError("Supply exactly one of signal_key or model_name")
        kind, value = (
            ("signal", signal_key) if signal_key is not None else ("model", model_name)
        )
        self._policies[kind, non_empty_string(value, kind)] = policy

    def register_executor(
        self,
        executor: FutureRollExecutor | None = None,
    ) -> FutureRollExecutor:
        """Register or reuse the process's one mode-specific executor.

        Args:
            executor: Optional custom executor. When omitted, the built-in
                implementation for the selected mode is created.

        Raises:
            RuntimeError: If no execution model has established the process mode.
            ValueError: If the executor's mode differs or another executor is bound.
        """

        mode = self.controller.execution_mode
        if mode is None:
            raise RuntimeError("Register an execution model before its roll executor")
        if self.executor is not None:
            if executor is not None and executor is not self.executor:
                raise ValueError("A different FutureRollExecutor is already registered")
            if self.executor.mode is not mode:
                raise ValueError(
                    "Registered roll executor does not match Controller execution mode"
                )
            return self.executor
        selected = executor or (
            ContractFutureRollExecutor()
            if mode is ExecutionMode.CONTRACT
            else SignalFutureRollExecutor()
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
        """Replace signal execution automatic-roll declarations."""

        self.future_roll_policies = dict(policies)

    def recover(self) -> bool:
        """Resume every non-complete persisted roll.

        Returns:
            True when recoverable roll state existed.
        """

        states = self.book.rolls.all(active_only=True)
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

    def roll(self, *args: object, now: datetime | None = None) -> None:
        """Check due rolls on the event loop, using the default or custom policy.

        Controller already schedules one daily process-lifetime check. Custom
        schedulers may call this method more often; do not add reconnect timers.
        ``now`` is an optional aware check time for deterministic policies/tests.
        In-flight work always resumes its persisted endpoints and stage.
        """
        if self._checking or not self.controller.broker_ready:
            return
        check_time = aware_datetime(now or datetime.now(timezone.utc), "now")
        self._checking = True
        try:
            self._check(check_time.astimezone(timezone.utc))
        finally:
            self._checking = False

    def _check(self, now: datetime) -> None:
        """Validate policy plans before initiating any newly discovered work."""

        executor = self.executor
        if executor is None:
            if self.book.rolls.all(active_only=True):
                self._required_executor()
            log.debug("No FutureRollExecutor is registered.")
            return
        self.recover()
        holdings = executor.holdings()
        if not holdings:
            log.debug("No futures positions require rolling.")
            return
        grouped = self._due_holdings(holdings, now)
        for series_key, by_contract in grouped.items():
            current = self.book.rolls.for_series(series_key)
            if current is not None and current.stage is not FutureRollStage.COMPLETE:
                continue
            endpoints = max(
                by_contract,
                key=lambda pair: (
                    pair[0].lastTradeDateOrContractMonth,
                    pair[0].conId,
                ),
            )
            old_contract, new_contract = endpoints
            planned = by_contract[endpoints]
            old_holdings = [holding for holding, _ in planned]
            if not self._broker_quantity_matches(old_contract):
                log.info(
                    "Roll deferred pending position reconciliation for %s",
                    old_contract.localSymbol,
                )
                self.controller.suspend_broker_work()
                self.controller.request_position_sync()
                return
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
            state = replace(
                state,
                occurrence_keys=tuple(
                    key
                    for holding, decision in planned
                    for key in self._occurrence_keys(holding, decision)
                ),
                completed_occurrences=current.completed_occurrences if current else (),
            )
            self.book.update_roll(state)
            log.warning(
                "Rolling futures series %s from %s to %s",
                series_key,
                old_contract.localSymbol,
                new_contract.localSymbol,
            )
            executor.advance(state)

    def _due_holdings(
        self, holdings: Sequence[RollHolding], now: datetime
    ) -> dict[
        str, dict[tuple[ibi.Future, ibi.Future], list[tuple[RollHolding, RollDecision]]]
    ]:
        """Group validated endpoints; never guess series membership from symbols."""
        grouped: dict[
            str,
            dict[tuple[ibi.Future, ibi.Future], list[tuple[RollHolding, RollDecision]]],
        ] = defaultdict(lambda: defaultdict(list))
        for holding in holdings:
            try:
                series_key = self.controller.contract_registry.series_key(
                    holding.contract
                )
                selector = self.controller.contract_registry.selector_for_series(
                    series_key
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.critical(
                    "Cannot identify futures series for conId=%s: %s",
                    holding.contract.conId,
                    exc,
                )
                continue
            current = self.book.rolls.for_series(series_key)
            if current is not None and current.stage is not FutureRollStage.COMPLETE:
                continue
            if not isinstance(selector, FutureSelector):
                raise TypeError("Futures rolling requires a FutureSelector")
            selector = replace(selector, today=now.replace(tzinfo=None))
            identity = (
                ("signal", holding.signal_key)
                if holding.signal_key is not None
                else ("model", holding.execution_model_name)
            )
            policy = self._policies.get(identity, self._default_policy)
            decision = policy.plan(holding, selector, now=now)
            if decision is None:
                continue
            if not isinstance(decision, RollDecision):
                raise TypeError(
                    "FutureRollPolicy.plan must return RollDecision or None"
                )
            keys = self._occurrence_keys(holding, decision)
            if keys and current and keys[0] in current.completed_occurrences:
                continue
            if (
                self.controller.contract_registry.series_key(decision.destination)
                != series_key
            ):
                raise ValueError("Roll destination must belong to the holding's series")
            if decision.destination.conId == holding.contract.conId:
                raise ValueError("Roll destination must differ from the held Contract")
            grouped[series_key][holding.contract, decision.destination].append(
                (holding, decision)
            )
        return grouped

    @staticmethod
    def _occurrence_keys(
        holding: RollHolding, decision: RollDecision
    ) -> tuple[str, ...]:
        """Mark replacement exposure too, preventing fixed-schedule cascades."""
        if decision.occurrence is None:
            return ()
        if holding.signal_key is not None:
            return (json.dumps(("signal", holding.signal_key, decision.occurrence)),)
        return tuple(
            json.dumps(("contract", con_id, decision.occurrence))
            for con_id in (holding.contract.conId, decision.destination.conId)
        )

    def _broker_quantity_matches(self, contract: ibi.Future) -> bool:
        return self.book.positions.filled_quantity(
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
                filled_quantity=holding.filled_quantity,
                signal_key=holding.signal_key,
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
            completed_occurrences=(
                previous.completed_occurrences
                if (previous := self.book.rolls.for_series(series_key)) is not None
                else ()
            ),
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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if not self._checking:
                self.roll()
        else:
            loop.call_soon(self.roll)

    def __repr__(self) -> str:
        return (
            f"FutureRoller(executor={self.executor!r}, "
            f"future_roll_policies={self.future_roll_policies!r})"
        )
