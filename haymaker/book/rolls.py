"""Durable futures-roll records; execution policy lives outside Book."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import ib_insync as ibi

from ..misc import decode_tree, tree
from ..validators import (
    aware_datetime,
    finite_number,
    non_empty_string,
    qualified_contract,
)
from ..saver import AbstractBaseSaver
from .persistence import utc_now, PersistenceWriter

from enum import StrEnum
from .targets import TargetState


class FutureRollMode(StrEnum):
    """Identify the process's mutually exclusive futures-roll accounting mode.

    ``DIRECT`` rolls account-wide concrete Contract targets.
    ``BRACKET`` rolls one-to-one episodes attributed by ``source_key`` and
    reinstalls their protective orders.
    """

    BRACKET = "BRACKET"
    DIRECT = "DIRECT"


class FutureRollStage(StrEnum):
    """Record durable progress through one recoverable futures-series roll.

    Values describe broker work as well as bracket-protection replacement.
    ``BLOCKED`` is intentionally durable and requires operator review;
    ``COMPLETE`` is the only state excluded from active recovery.
    """

    PLANNED = "PLANNED"
    WAITING_FOR_ACTIVE_WORK = "WAITING_FOR_ACTIVE_WORK"
    ROLL_ORDER_ACTIVE = "ROLL_ORDER_ACTIVE"
    ROLL_FILLED = "ROLL_FILLED"
    LOADING_REFERENCE_PRICE = "LOADING_REFERENCE_PRICE"
    CANCELLING_PROTECTION = "CANCELLING_PROTECTION"
    INSTALLING_STOP = "INSTALLING_STOP"
    INSTALLING_TAKE_PROFIT = "INSTALLING_TAKE_PROFIT"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, kw_only=True)
class RollParticipant:
    """Persist one logical holding participating in a futures-series roll.

    Args:
        execution_model_name: Stable target-model name that owns the holding.
        quantity: Signed quantity attributed to this participant. A zero
            non-trading source records an episode that closed while waiting.
        source_key: One-to-one identity in bracket mode.
        position_id: Optional one-to-one position episode.
        requires_trade: Whether this participant supplies the physical BAG
            order. Offset logical sources may require only price adjustment and
            protection replacement.

    A source identifies a one-to-one episode; absence denotes direct mode.
    """

    execution_model_name: str
    quantity: float
    source_key: str | None = None
    position_id: str | None = None
    requires_trade: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_model_name",
            non_empty_string(self.execution_model_name, "execution_model_name"),
        )
        if self.source_key is not None:
            object.__setattr__(
                self, "source_key", non_empty_string(self.source_key, "source_key")
            )
        object.__setattr__(self, "quantity", finite_number(self.quantity, "quantity"))
        if not self.quantity and (self.source_key is None or self.requires_trade):
            raise ValueError("Zero roll quantity is only valid for a skipped source")
        if not isinstance(self.requires_trade, bool):
            raise TypeError("requires_trade must be a bool")

    def encode(self) -> dict[str, Any]:
        """Serialize one roll participant without losing episode attribution."""
        return {
            "execution_model_name": self.execution_model_name,
            "quantity": self.quantity,
            "source_key": self.source_key,
            "position_id": self.position_id,
            "requires_trade": self.requires_trade,
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> RollParticipant:
        """Restore and validate a persisted participant."""
        if "target_key" in data:
            raise ValueError("Old keyed roll schema requires standalone conversion")
        return cls(
            execution_model_name=str(data["execution_model_name"]),
            quantity=float(data["quantity"]),
            source_key=data.get("source_key"),
            position_id=data.get("position_id"),
            requires_trade=bool(data.get("requires_trade", True)),
        )


@dataclass(frozen=True, kw_only=True)
class RollState:
    """Recover the current roll operation for one registered futures series.

    Args:
        series_key: ContractRegistry identity shared by all qualified expiries
            of one registered blueprint.
        mode: Exclusive direct or bracket accounting mode.
        executor_name: Stable FutureRollExecutor recovery identity.
        old_contract: Concrete held Future being left.
        new_contract: Concrete policy-selected Future being entered.
        participants: Frozen logical holdings included in this operation.
        target_transfers: Idempotent concrete target snapshots for direct rolls.
        occurrence_keys: Schedule markers to record upon successful completion.
        completed_occurrences: Previously completed schedule markers retained
            when the next operation replaces this series' current record.
        participant_index: Durable cursor for serial participant processing.
        stage: Last durably accepted roll stage.
        roll_order_id: Current BAG order id, when one is active or filled.
        old_protection_order_ids: Bracket orders being replaced.
        replacement_stop_order_id: New critical stop order id.
        replacement_take_profit_order_id: New optional take-profit order id.
        reference_price: Filled or observed calendar-spread price used to shift
            protective prices.
        failure_reason: Actionable explanation when ``stage`` is ``BLOCKED``.
        created_at: Start time of this roll generation.
        updated_at: Time of the latest durable transition.

    Book stores one current record per ``series_key``. Framework executors
    normally construct and advance it; custom executors must persist each stage
    before performing its broker side effect.
    """

    series_key: str
    mode: FutureRollMode
    executor_name: str
    old_contract: ibi.Future
    new_contract: ibi.Future
    participants: Sequence[RollParticipant]
    target_transfers: Sequence[TargetState] = ()
    occurrence_keys: Sequence[str] = ()
    completed_occurrences: Sequence[str] = ()
    participant_index: int = 0
    stage: FutureRollStage = FutureRollStage.PLANNED
    roll_order_id: int | None = None
    old_protection_order_ids: Sequence[int] = ()
    replacement_stop_order_id: int | None = None
    replacement_take_profit_order_id: int | None = None
    reference_price: float | None = None
    failure_reason: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "series_key", non_empty_string(self.series_key, "series_key")
        )
        object.__setattr__(
            self,
            "executor_name",
            non_empty_string(self.executor_name, "executor_name"),
        )
        if not isinstance(self.mode, FutureRollMode):
            raise TypeError("mode must be a FutureRollMode")
        if not isinstance(self.stage, FutureRollStage):
            raise TypeError("stage must be a FutureRollStage")
        if not isinstance(self.old_contract, ibi.Future) or not isinstance(
            self.new_contract, ibi.Future
        ):
            raise TypeError("old_contract and new_contract must be Futures")
        qualified_contract(self.old_contract, "old_contract")
        qualified_contract(self.new_contract, "new_contract")
        if self.old_contract.conId == self.new_contract.conId:
            raise ValueError("old_contract and new_contract must differ")
        participants = tuple(self.participants)
        if not participants or not all(
            isinstance(participant, RollParticipant) for participant in participants
        ):
            raise ValueError("participants must contain RollParticipant values")
        object.__setattr__(self, "participants", participants)
        if any(
            (p.source_key is None) != (self.mode is FutureRollMode.DIRECT)
            for p in participants
        ):
            raise ValueError("Roll participant attribution does not match mode")
        transfers = tuple(self.target_transfers)
        if not all(isinstance(target, TargetState) for target in transfers):
            raise TypeError("target_transfers must contain TargetState values")
        object.__setattr__(self, "target_transfers", transfers)
        for name in ("occurrence_keys", "completed_occurrences"):
            keys = tuple(non_empty_string(key, name) for key in getattr(self, name))
            object.__setattr__(self, name, tuple(dict.fromkeys(keys)))
        if not isinstance(self.participant_index, int) or isinstance(
            self.participant_index, bool
        ):
            raise TypeError("participant_index must be an integer")
        if not 0 <= self.participant_index <= len(participants):
            raise ValueError("participant_index is outside participants")
        old_order_ids = tuple(self.old_protection_order_ids)
        if not all(
            isinstance(order_id, int) and order_id > 0 for order_id in old_order_ids
        ):
            raise ValueError("old_protection_order_ids must be positive integers")
        object.__setattr__(self, "old_protection_order_ids", old_order_ids)
        for name in (
            "roll_order_id",
            "replacement_stop_order_id",
            "replacement_take_profit_order_id",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be None or a positive integer")
        if self.reference_price is not None:
            object.__setattr__(
                self,
                "reference_price",
                finite_number(self.reference_price, "reference_price"),
            )
        if self.failure_reason is not None:
            object.__setattr__(
                self,
                "failure_reason",
                non_empty_string(self.failure_reason, "failure_reason"),
            )
        aware_datetime(self.created_at, "created_at")
        aware_datetime(self.updated_at, "updated_at")

    @property
    def current_participant(self) -> RollParticipant | None:
        """Return the participant at the durable cursor, if any."""

        if self.participant_index == len(self.participants):
            return None
        return self.participants[self.participant_index]

    @property
    def terminal(self) -> bool:
        """Return whether automatic work must no longer advance."""

        return self.stage in {FutureRollStage.COMPLETE, FutureRollStage.BLOCKED}

    def encode(self) -> dict[str, Any]:
        """Serialize the complete roll checkpoint and nested recovery records."""
        return {
            "state_key": f"roll:{self.series_key}",
            "state_type": "roll",
            "series_key": self.series_key,
            "mode": self.mode.value,
            "executor_name": self.executor_name,
            "old_contract": tree(self.old_contract),
            "new_contract": tree(self.new_contract),
            "participants": [participant.encode() for participant in self.participants],
            "target_transfers": [t.encode() for t in self.target_transfers],
            "participant_index": self.participant_index,
            "occurrence_keys": list(self.occurrence_keys),
            "completed_occurrences": list(self.completed_occurrences),
            "stage": self.stage.value,
            "roll_order_id": self.roll_order_id,
            "old_protection_order_ids": list(self.old_protection_order_ids),
            "replacement_stop_order_id": self.replacement_stop_order_id,
            "replacement_take_profit_order_id": (self.replacement_take_profit_order_id),
            "reference_price": self.reference_price,
            "failure_reason": self.failure_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> RollState:
        """Restore a current roll checkpoint, including its pending transfers."""
        return cls(
            series_key=str(data["series_key"]),
            mode=FutureRollMode(str(data["mode"])),
            executor_name=str(data["executor_name"]),
            old_contract=decode_tree(data["old_contract"]),
            new_contract=decode_tree(data["new_contract"]),
            participants=tuple(
                RollParticipant.decode(participant)
                for participant in data["participants"]
            ),
            target_transfers=tuple(
                TargetState.decode(t) for t in data.get("target_transfers", ())
            ),
            participant_index=int(data.get("participant_index", 0)),
            occurrence_keys=tuple(data.get("occurrence_keys", ())),
            completed_occurrences=tuple(data.get("completed_occurrences", ())),
            stage=FutureRollStage(str(data["stage"])),
            roll_order_id=data.get("roll_order_id"),
            old_protection_order_ids=tuple(data.get("old_protection_order_ids", ())),
            replacement_stop_order_id=data.get("replacement_stop_order_id"),
            replacement_take_profit_order_id=data.get(
                "replacement_take_profit_order_id"
            ),
            reference_price=data.get("reference_price"),
            failure_reason=data.get("failure_reason"),
            created_at=decode_tree(data["created_at"]),
            updated_at=decode_tree(data["updated_at"]),
        )


class RollStore:
    """Own per-series recovery checkpoints, not roll policy or broker sequencing."""

    def __init__(self, saver: AbstractBaseSaver, writer: PersistenceWriter) -> None:
        """Share Book's ordered state persistence."""
        self._saver = saver
        self._writer = writer
        self._items: dict[str, RollState] = {}

    def _put(self, state: RollState) -> None:
        """Persist a roll checkpoint before dependent accounting changes."""
        self._writer.save(self._saver, state.encode())
        self._items[state.series_key] = state

    def _restore(self, document: Mapping[str, Any]) -> None:
        """Load a checkpoint without advancing its execution stage."""
        state = RollState.decode(document)
        self._items[state.series_key] = state

    def _clear(self, cleared_at: datetime) -> None:
        """Persist completed tombstones during an explicit account reset."""
        for state in self._items.values():
            self._put(
                replace(
                    state,
                    participant_index=len(state.participants),
                    stage=FutureRollStage.COMPLETE,
                    roll_order_id=None,
                    old_protection_order_ids=(),
                    replacement_stop_order_id=None,
                    replacement_take_profit_order_id=None,
                    failure_reason=None,
                    updated_at=cleared_at,
                )
            )
        self._items.clear()

    def for_series(self, series_key: str) -> RollState | None:
        """Return the current persisted roll for one futures series."""

        return self._items.get(series_key)

    def all(self, *, active_only: bool = False) -> tuple[RollState, ...]:
        """Return all current series rolls, optionally excluding completed ones."""

        return tuple(
            state
            for state in self._items.values()
            if not active_only or state.stage is not FutureRollStage.COMPLETE
        )

    def for_source(self, source_key: str) -> RollState | None:
        """Return a non-complete roll containing one one-to-one source."""

        return next(
            (
                state
                for state in self._items.values()
                if state.stage is not FutureRollStage.COMPLETE
                and any(
                    participant.source_key == source_key
                    for participant in state.participants
                )
            ),
            None,
        )

    def for_contract(self, contract: ibi.Contract) -> RollState | None:
        """Return an active direct roll reserving either concrete endpoint."""
        con_id = qualified_contract(contract).conId
        return next(
            (
                state
                for state in self._items.values()
                if not state.terminal
                and state.mode is FutureRollMode.DIRECT
                and con_id in (state.old_contract.conId, state.new_contract.conId)
            ),
            None,
        )

    def active_bracket(self) -> tuple[RollState, ...]:
        """Return incomplete one-to-one rolls that own broker projections."""

        return tuple(
            state
            for state in self._items.values()
            if state.mode is FutureRollMode.BRACKET
            and state.stage is not FutureRollStage.COMPLETE
        )
