"""Dataloader runtime settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from haymaker.datastore import AbstractBaseStore
from haymaker.validators import bar_size_validator, wts_validator

from . import helpers
from .pacer import RequestPacing


@dataclass(frozen=True)
class DataloaderSettings:
    """Non-connection settings used by one dataloader runtime session."""

    bar_size: str
    wts: str
    max_bars: int
    max_period: int
    fill_gaps: bool
    auto_save_interval: int
    number_of_workers: int
    store: AbstractBaseStore
    source: str
    pacer_restrictions: list[tuple[int, float]]
    pacer_no_restriction: bool
    pacer_allowance_fraction: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DataloaderSettings":
        """Create dataloader settings from a dataloader config mapping."""

        pacer_allowance_fraction = config.get("pacer_allowance_fraction", 1.0)
        if not 0 < pacer_allowance_fraction <= 1:
            raise ValueError("pacer_allowance_fraction must be > 0 and <= 1")

        return cls(
            bar_size=bar_size_validator(config["barSize"]),
            wts=wts_validator(config["wts"]),
            max_bars=config.get("max_bars", 50000),
            max_period=config.get("max_period", 30),
            fill_gaps=config.get("fill_gaps", True),
            auto_save_interval=config.get("auto_save_interval", 0),
            number_of_workers=config.get("number_of_workers", 20),
            store=config["datastore"],
            source=config["source"],
            pacer_restrictions=list(config.get("pacer_restrictions", [])),
            pacer_no_restriction=config.get("pacer_no_restriction", False),
            pacer_allowance_fraction=pacer_allowance_fraction,
        )

    def normalize_datetime(self, value: date | datetime) -> date | datetime:
        """Normalize a date or datetime using this session's bar size."""

        return helpers.datetime_normalizer(value, barsize=self.bar_size)

    def create_pacing(self) -> RequestPacing:
        """Create request pacing state for this dataloader session."""

        return RequestPacing(
            self.bar_size,
            self.wts,
            restrictions=self.pacer_restrictions,
            no_restriction=self.pacer_no_restriction,
        )
