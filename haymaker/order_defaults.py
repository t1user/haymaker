"""Default Interactive Brokers order fields used by execution models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import ib_insync as ibi


def _order_mapping(value: object, name: str) -> dict[str, Any]:
    """Return one reconstructed and broker-compatible order mapping.

    Args:
        value: Plain mapping loaded from framework configuration.
        name: Order-default group used in failure messages.

    Returns:
        A copied mapping with algorithm parameters converted to ``TagValue``.

    Raises:
        TypeError: If the mapping cannot construct an IB order.
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"orders.{name} must be a mapping")
    order = dict(value)
    if "algoParams" in order:
        params = order["algoParams"]
        if not isinstance(params, list):
            raise TypeError(f"orders.{name}.algoParams must be a list")
        order["algoParams"] = [
            item if isinstance(item, ibi.TagValue) else ibi.TagValue(**dict(item))
            for item in params
        ]
    ibi.Order(**order)
    return order


@dataclass(frozen=True)
class OrderDefaults:
    """IB order defaults shared by execution models."""

    open: Mapping[str, Any] = field(default_factory=dict)
    close: Mapping[str, Any] = field(default_factory=dict)
    stop: Mapping[str, Any] = field(default_factory=dict)
    take_profit: Mapping[str, Any] = field(default_factory=dict)
    oca_type: int = 1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Construct defaults from one plain configuration section.

        Args:
            values: Merged ``orders`` configuration section.

        Returns:
            Reconstructed order defaults ready for execution models.
        """

        options = dict(values)
        for name in ("open", "close", "stop", "take_profit"):
            if name in options:
                options[name] = _order_mapping(options[name], name)
        result = cls(**options)
        if result.oca_type <= 0:
            raise ValueError("orders.oca_type must be positive")
        return result
