"""Latest contract targets and concrete-contract recovery identities."""

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
from .schema import require_schema
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .positions import PositionStore, SignalTarget

from .persistence import utc_now, PersistenceWriter


@dataclass(frozen=True, kw_only=True)
class ContractTarget:
    """Accepted desired exposure for one concrete Contract.

    Args:
        execution_model_name: Owner responsible for converging this contract.
        contract: Qualified contract identifying one concrete conId.
        target_quantity: Absolute signed desired exposure, excluding working orders.
        target_created_at: Accepted target message time used for supersession.
        updated_at: Time this accepted record was last updated.
    """

    execution_model_name: str
    contract: ibi.Contract
    target_quantity: float
    target_created_at: datetime
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
        aware_datetime(self.updated_at, "updated_at")

    def encode(self) -> dict[str, Any]:
        """Serialize accepted target fields for the shared recovery document."""
        return {
            "schema_version": 1,
            "state_key": f"contract_execution:{self.contract.conId}",
            "state_type": "contract_execution",
            "execution_model_name": self.execution_model_name,
            "conId": self.contract.conId,
            "contract": tree(self.contract),
            "target_quantity": self.target_quantity,
            "target_created_at": self.target_created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> ContractTarget:
        """Restore a concrete target; old keyed schemas require conversion."""
        require_schema(data)
        if data.get("state_key") != f"contract_execution:{data['conId']}":
            raise ValueError("Old keyed target schema requires standalone conversion")
        target = cls(
            execution_model_name=str(data["execution_model_name"]),
            contract=decode_tree(data["contract"]),
            target_quantity=float(data["target_quantity"]),
            target_created_at=decode_tree(data["target_created_at"]),
            updated_at=decode_tree(data["updated_at"]),
        )

        if target.contract.conId != data["conId"]:
            raise ValueError("Contract target identity does not match its Contract")
        return target


@dataclass(frozen=True, kw_only=True)
class ContractExecutionState:
    """Keep a target, its reset cutoff and tombstone in one atomic document."""

    target: ContractTarget
    fill_evidence_start_at: datetime | None = None
    cleared: bool = False

    def __post_init__(self) -> None:
        if self.fill_evidence_start_at is not None:
            aware_datetime(self.fill_evidence_start_at, "fill_evidence_start_at")

    def encode(self) -> dict[str, Any]:
        return {
            **self.target.encode(),
            "fill_evidence_start_at": self.fill_evidence_start_at,
            "cleared": self.cleared,
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> ContractExecutionState:
        return cls(
            target=ContractTarget.decode(data),
            fill_evidence_start_at=decode_tree(data.get("fill_evidence_start_at")),
            cleared=bool(data.get("cleared", False)),
        )


class TargetStore:
    """Query accepted targets; signal records share the position-state owner."""

    def __init__(
        self,
        saver: AbstractBaseSaver,
        writer: PersistenceWriter,
        positions: PositionStore,
    ) -> None:
        self._saver = saver
        self._writer = writer
        self._positions = positions
        self._items: dict[int, ContractExecutionState] = {}

    @property
    def cutoffs(self) -> Mapping[int, datetime]:
        """Expose read-only reset boundaries used by position accounting."""
        return MappingProxyType(
            {
                key: state.fill_evidence_start_at
                for key, state in self._items.items()
                if state.fill_evidence_start_at is not None
            }
        )

    def for_signal(self, signal_key: str) -> SignalTarget | None:
        """Return the accepted signal target, if one exists."""
        state = self._positions._for_signal(signal_key)
        return state.target if state is not None else None

    def for_contract(self, contract: ibi.Contract) -> ContractTarget | None:
        """Return the latest setpoint for one qualified concrete Contract."""
        state = self._items.get(qualified_contract(contract).conId)
        return state.target if state is not None and not state.cleared else None

    def _put(self, target: ContractTarget) -> ContractTarget:
        current = self.for_contract(target.contract)
        if current is not None and target.target_created_at < current.target_created_at:
            return current
        state = ContractExecutionState(
            target=target,
            fill_evidence_start_at=self.cutoffs.get(target.contract.conId),
        )
        self._writer.save(self._saver, state.encode())
        self._items[target.contract.conId] = state
        return target

    def _restore(self, document: Mapping[str, Any]) -> None:
        state = ContractExecutionState.decode(document)
        self._items[state.target.contract.conId] = state

    def _clear(self, contracts: Collection[ibi.Contract], cleared_at: datetime) -> None:
        targets = {target.contract.conId: target for target in self.all()}
        for contract in contracts:
            targets.setdefault(
                contract.conId,
                ContractTarget(
                    execution_model_name="reset",
                    contract=contract,
                    target_quantity=0,
                    target_created_at=cleared_at,
                ),
            )
        for target in targets.values():
            state = ContractExecutionState(
                target=replace(
                    target,
                    target_quantity=0,
                    target_created_at=cleared_at,
                    updated_at=cleared_at,
                ),
                fill_evidence_start_at=cleared_at,
                cleared=True,
            )
            self._writer.save(self._saver, state.encode())
            self._items[target.contract.conId] = state

    def all(
        self, execution_model_name: str | None = None
    ) -> tuple[ContractTarget, ...]:
        """Return accepted contract targets, optionally restricted by owner."""
        return tuple(
            state.target
            for state in self._items.values()
            if not state.cleared
            and (
                execution_model_name is None
                or state.target.execution_model_name == execution_model_name
            )
        )
