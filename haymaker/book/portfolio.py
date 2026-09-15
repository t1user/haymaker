"""Opaque custom-Portfolio recovery mappings and their persistence."""

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ..misc import tree, decode_tree
from ..validators import non_empty_string, readonly_mapping
from ..saver import AbstractBaseSaver
from .persistence import PersistenceWriter, utc_now


class PortfolioStateStore:
    """Persist user-defined normalized mappings without interpreting allocation policy."""

    def __init__(self, saver: AbstractBaseSaver, writer: PersistenceWriter) -> None:
        """Use the existing state collection and Book's shared writer."""
        self._saver = saver
        self._writer = writer
        self._items: dict[str, Mapping[str, Any]] = {}

    def _restore(self, document: Mapping[str, Any]) -> None:
        """Restore one opaque mapping under its validated natural key."""
        key = non_empty_string(document["portfolio_key"], "portfolio_key")
        self._items[key] = readonly_mapping(
            decode_tree(document.get("state", {})), "state"
        )

    def _clear(self, cleared_at: datetime) -> None:
        """Persist empty mappings before forgetting in-memory portfolio state."""
        for key in self._items:
            self._writer.save(
                self._saver,
                {
                    "state_key": f"portfolio:{key}",
                    "state_type": "portfolio",
                    "portfolio_key": key,
                    "state": {},
                    "updated_at": cleared_at,
                },
            )
        self._items.clear()

    def save(self, portfolio_key: str, state: Mapping[str, Any]) -> None:
        """Persist one Portfolio's normalized recovery mapping."""

        portfolio_key = non_empty_string(portfolio_key, "portfolio_key")
        copied = readonly_mapping(state, "state")
        self._writer.save(
            self._saver,
            {
                "state_key": f"portfolio:{portfolio_key}",
                "state_type": "portfolio",
                "portfolio_key": portfolio_key,
                "state": tree(dict(copied)),
                "updated_at": utc_now(),
            },
        )
        self._items[portfolio_key] = copied

    def load(self, portfolio_key: str) -> Mapping[str, Any] | None:
        """Load one Portfolio recovery mapping."""

        return self._items.get(non_empty_string(portfolio_key, "portfolio_key"))
