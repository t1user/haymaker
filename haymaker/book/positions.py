"""Source episodes and shared concrete-contract position accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

import ib_insync as ibi

from ..misc import decode_tree, tree
from ..validators import (
    aware_datetime,
    finite_number,
    ib_contract,
    non_empty_string,
    qualified_contract,
    readonly_mapping,
)
from .persistence import utc_now
from .orders import FillRecord, OrderInfo, fill_direction


@dataclass(frozen=True, kw_only=True)
class PositionState:
    """Recover an episode separately from its latest accepted target.

    ``contract`` and ``bracket_inputs`` belong to the holding or submitted
    entry. ``target_contract`` and ``target_bracket_inputs`` belong to the
    pending opening destination; accepting a reversal must not change the
    Contract or protection inputs of the episode being closed.
    """

    source_key: str
    execution_model_name: str
    contract: ibi.Contract | None = None
    quantity: float = 0.0
    target_quantity: float | None = None
    target_created_at: datetime | None = None
    target_contract: ibi.Contract | None = None
    target_bracket_inputs: Mapping[str, Any] = field(default_factory=dict)
    position_id: str | None = None
    blocked_direction: Literal[-1, 1] | None = None
    bracket_inputs: Mapping[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=utc_now)
    _applied_fill_keys: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_applied_fill_keys", frozenset(self._applied_fill_keys)
        )
        object.__setattr__(
            self, "source_key", non_empty_string(self.source_key, "source_key")
        )
        object.__setattr__(
            self,
            "execution_model_name",
            non_empty_string(self.execution_model_name, "execution_model_name"),
        )
        if self.contract is not None:
            ib_contract(self.contract)
        if self.target_contract is not None:
            ib_contract(self.target_contract)
        object.__setattr__(
            self,
            "target_bracket_inputs",
            readonly_mapping(self.target_bracket_inputs, "target_bracket_inputs"),
        )
        if isinstance(self.blocked_direction, bool) or (
            self.blocked_direction not in (None, -1, 1)
        ):
            raise ValueError("blocked_direction must be None, -1, or 1")
        object.__setattr__(self, "quantity", finite_number(self.quantity, "quantity"))
        if self.target_quantity is not None:
            object.__setattr__(
                self,
                "target_quantity",
                finite_number(self.target_quantity, "target_quantity"),
            )
        if self.target_created_at is not None:
            aware_datetime(self.target_created_at, "target_created_at")
        aware_datetime(self.updated_at, "updated_at")
        object.__setattr__(
            self,
            "bracket_inputs",
            readonly_mapping(self.bracket_inputs, "bracket_inputs"),
        )

    def with_fill(self, info: OrderInfo, record: FillRecord) -> PositionState | None:
        """Return the episode after one unapplied fill, or None for a duplicate."""
        if record.deduplication_key in self._applied_fill_keys:
            return None
        old_quantity = self.quantity
        quantity = old_quantity + record.execution.shares * fill_direction(record)
        blocked = self.blocked_direction
        if info.role in {"STOP_LOSS", "TAKE_PROFIT"} and old_quantity and quantity == 0:
            blocked = 1 if old_quantity > 0 else -1
        elif info.role == "OPEN" and quantity != 0:
            blocked = None
        episode_closed = quantity == 0 and info.role in {
            "CLOSE",
            "STOP_LOSS",
            "TAKE_PROFIT",
        }
        protective_exit = quantity == 0 and info.role in {
            "STOP_LOSS",
            "TAKE_PROFIT",
        }
        target_quantity = 0.0 if protective_exit else self.target_quantity
        return replace(
            self,
            contract=info.trade.contract,
            quantity=quantity,
            target_quantity=target_quantity,
            target_contract=None if protective_exit else self.target_contract,
            target_bracket_inputs=(
                {} if protective_exit else self.target_bracket_inputs
            ),
            blocked_direction=blocked,
            position_id=None if episode_closed else self.position_id,
            bracket_inputs=(
                {} if episode_closed and not target_quantity else self.bracket_inputs
            ),
            updated_at=utc_now(),
            _applied_fill_keys=self._applied_fill_keys | {record.deduplication_key},
        )

    def corrected(self, quantity: float, corrected_at: datetime) -> PositionState:
        """Align an episode and its pending target to an accepted broker correction.

        Choosing the source and quantity belongs to reconciliation. Flat
        corrections clear episode inputs, but preserve the direction block and
        the fill checkpoint so historical evidence cannot undo the correction.
        """
        flat = quantity == 0
        return replace(
            self,
            quantity=quantity,
            target_quantity=quantity,
            target_contract=None if flat else self.contract,
            target_bracket_inputs={} if flat else self.bracket_inputs,
            target_created_at=corrected_at,
            position_id=None if flat else self.position_id,
            bracket_inputs={} if flat else self.bracket_inputs,
            updated_at=corrected_at,
        )

    def encode(self) -> dict[str, Any]:
        """Serialize the episode and its applied-fill checkpoint together."""
        return {
            "state_key": f"position:{self.source_key}",
            "state_type": "position",
            "source_key": self.source_key,
            "execution_model_name": self.execution_model_name,
            "contract": tree(self.contract),
            "quantity": self.quantity,
            "target_quantity": self.target_quantity,
            "target_created_at": self.target_created_at,
            "target_contract": tree(self.target_contract),
            "target_bracket_inputs": tree(dict(self.target_bracket_inputs)),
            "position_id": self.position_id,
            "blocked_direction": self.blocked_direction,
            "bracket_inputs": tree(dict(self.bracket_inputs)),
            "updated_at": self.updated_at,
            "applied_fill_keys": sorted(self._applied_fill_keys),
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> PositionState:
        """Restore checkpointed state; uncheckpointed schemas need conversion."""
        if "applied_fill_keys" not in data:
            raise ValueError(
                "Position state lacks a fill checkpoint; standalone conversion required"
            )
        return cls(
            source_key=str(data["source_key"]),
            execution_model_name=str(data["execution_model_name"]),
            contract=decode_tree(data.get("contract")),
            quantity=float(data.get("quantity", 0.0)),
            target_quantity=data.get("target_quantity"),
            target_created_at=decode_tree(data.get("target_created_at")),
            target_contract=decode_tree(data["target_contract"]),
            target_bracket_inputs=decode_tree(data["target_bracket_inputs"]),
            position_id=data.get("position_id"),
            blocked_direction=data.get("blocked_direction"),
            bracket_inputs=decode_tree(data.get("bracket_inputs", {})),
            updated_at=decode_tree(data["updated_at"]),
            _applied_fill_keys=frozenset(data["applied_fill_keys"]),
        )


@dataclass(frozen=True, kw_only=True)
class ContractPosition:
    """Persist an accounted net balance, independently of execution mode.

    Book updates this balance with the underlying episode, order or roll
    mutation. Recovery verifies it against those records before trading starts.
    It is neither a desired target nor a broker-position snapshot.
    """

    contract: ibi.Contract
    quantity: float
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate a concrete Contract and finite signed balance."""
        qualified_contract(self.contract)
        object.__setattr__(self, "quantity", finite_number(self.quantity, "quantity"))
        aware_datetime(self.updated_at, "updated_at")

    def encode(self) -> dict[str, Any]:
        """Serialize one maintained contract balance."""
        return {
            "state_key": f"balance:{self.contract.conId}",
            "state_type": "balance",
            "conId": self.contract.conId,
            "contract": tree(self.contract),
            "quantity": self.quantity,
            "updated_at": self.updated_at,
        }

    @classmethod
    def decode(cls, document: Mapping[str, Any]) -> ContractPosition:
        """Validate the concrete identity of a persisted balance."""
        balance = cls(
            contract=decode_tree(document["contract"]),
            quantity=document["quantity"],
            updated_at=decode_tree(document["updated_at"]),
        )
        if (
            document.get("state_key") != f"balance:{balance.contract.conId}"
            or document.get("conId") != balance.contract.conId
        ):
            raise ValueError("Persisted balance identity does not match its Contract")
        return balance
