"""Latest direct targets and concrete-contract recovery identities."""

from __future__ import annotations

from collections.abc import Mapping, Collection
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
from types import MappingProxyType
from ..saver import AbstractBaseSaver
from .persistence import utc_now, PersistenceWriter


@dataclass(frozen=True, kw_only=True)
class TargetState:
    """Recover the latest shared direct-execution target.

    ``fill_evidence_start_at`` is Book-owned reset state. It preserves order
    and Fill history while excluding evidence that predates an explicit
    account-state clear from the rebuilt direct position.
    """

    execution_model_name: str
    contract: ibi.Contract
    target_quantity: float
    target_created_at: datetime
    fill_evidence_start_at: datetime | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_model_name",
            non_empty_string(self.execution_model_name, "execution_model_name"),
        )
        qualified_contract(self.contract)
        object.__setattr__(
            self,
            "target_quantity",
            finite_number(self.target_quantity, "target_quantity"),
        )
        aware_datetime(self.target_created_at, "target_created_at")
        if self.fill_evidence_start_at is not None:
            aware_datetime(
                self.fill_evidence_start_at,
                "fill_evidence_start_at",
            )
        aware_datetime(self.updated_at, "updated_at")

    def encode(self) -> dict[str, Any]:
        """Serialize a concrete target and its reset cutoff."""
        return {
            "state_key": f"target:{self.contract.conId}",
            "state_type": "target",
            "execution_model_name": self.execution_model_name,
            "conId": self.contract.conId,
            "contract": tree(self.contract),
            "target_quantity": self.target_quantity,
            "target_created_at": self.target_created_at,
            "fill_evidence_start_at": self.fill_evidence_start_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> TargetState:
        """Restore a concrete target; old keyed schemas require conversion."""
        if "target_key" in data or data.get("state_key") != f"target:{data['conId']}":
            raise ValueError("Old keyed target schema requires standalone conversion")
        return cls(
            execution_model_name=str(data["execution_model_name"]),
            contract=decode_tree(data["contract"]),
            target_quantity=float(data["target_quantity"]),
            target_created_at=decode_tree(data["target_created_at"]),
            fill_evidence_start_at=decode_tree(data.get("fill_evidence_start_at")),
            updated_at=decode_tree(data["updated_at"]),
        )


class TargetStore:
    """Own latest concrete targets and durable reset cutoffs, not allocation policy."""

    def __init__(self, saver: AbstractBaseSaver, writer: PersistenceWriter) -> None:
        """Share Book's state collection and ordered writer."""
        self._saver = saver
        self._writer = writer
        self._items: dict[int, TargetState] = {}
        self._cutoffs: dict[int, datetime] = {}

    @property
    def cutoffs(self) -> Mapping[int, datetime]:
        """Expose read-only reset boundaries used by position accounting."""
        return MappingProxyType(self._cutoffs)

    def for_contract(self, contract: ibi.Contract) -> TargetState | None:
        """Return the latest setpoint for one qualified concrete Contract."""
        return self._items.get(qualified_contract(contract).conId)

    def _put(self, state: TargetState) -> TargetState:
        """Persist an accepted latest target without overwriting newer decisions."""
        current = self._items.get(state.contract.conId)
        if current is not None and state.target_created_at < current.target_created_at:
            return current
        cutoff = self._cutoffs.get(state.contract.conId)
        if cutoff is not None:
            state = replace(state, fill_evidence_start_at=cutoff)
        self._writer.save(self._saver, state.encode())
        self._items[state.contract.conId] = state
        if state.fill_evidence_start_at is not None:
            self._cutoffs[state.contract.conId] = state.fill_evidence_start_at
        return state

    def _restore(self, document: Mapping[str, Any]) -> None:
        """Restore a target or its cleared tombstone without issuing live writes."""
        state = TargetState.decode(document)
        if state.fill_evidence_start_at is not None:
            self._cutoffs[state.contract.conId] = state.fill_evidence_start_at
        if not document.get("cleared", False):
            self._items[state.contract.conId] = state

    def _clear(self, contracts: Collection[ibi.Contract], cleared_at: datetime) -> None:
        """Persist tombstones, including residual exposure without a prior target."""
        targets = dict(self._items)
        for contract in contracts:
            targets.setdefault(
                contract.conId,
                TargetState(
                    execution_model_name="reset",
                    contract=contract,
                    target_quantity=0,
                    target_created_at=cleared_at,
                ),
            )
        for target in targets.values():
            document = replace(
                target,
                target_quantity=0,
                target_created_at=cleared_at,
                fill_evidence_start_at=cleared_at,
                updated_at=cleared_at,
            ).encode()
            document["cleared"] = True
            self._writer.save(self._saver, document)
            self._cutoffs[target.contract.conId] = cleared_at
        self._items.clear()

    def all(self, execution_model_name: str | None = None) -> tuple[TargetState, ...]:
        """Return recovered direct targets, optionally restricted by owner."""

        return tuple(
            state
            for state in self._items.values()
            if execution_model_name is None
            or state.execution_model_name == execution_model_name
        )
