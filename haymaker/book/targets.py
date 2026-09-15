"""Latest direct targets and concrete-contract recovery identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
from .persistence import utc_now


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
