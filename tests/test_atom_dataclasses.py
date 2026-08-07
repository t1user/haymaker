import inspect
from dataclasses import is_dataclass

import pytest

from haymaker.base import Atom
from haymaker.components import (
    BarAggregator,
    FuturesPandasAggregator,
    HistoricalDataStreamer,
    MktDataStreamer,
    PandasSignalModel,
    RealTimeBarsStreamer,
    SignalModel,
    TickByTickStreamer,
    VolumeGrouper,
)
from haymaker.controller import Controller


@pytest.mark.parametrize(
    "atom_type",
    (
        Controller,
        FuturesPandasAggregator,
        HistoricalDataStreamer,
        MktDataStreamer,
        PandasSignalModel,
        RealTimeBarsStreamer,
        SignalModel,
        TickByTickStreamer,
        VolumeGrouper,
    ),
)
def test_dataclass_atoms_preserve_identity_equality(atom_type: type[Atom]) -> None:
    """Dataclass components remain distinct stateful graph nodes."""

    assert is_dataclass(atom_type)
    assert atom_type.__eq__ is object.__eq__
    assert atom_type.__hash__ is object.__hash__


@pytest.mark.parametrize("aggregator_type", (BarAggregator, FuturesPandasAggregator))
def test_aggregator_onStart_matches_atom_interface(
    aggregator_type: type[Atom],
) -> None:
    """Aggregator startup overrides preserve the complete Atom signature."""

    assert inspect.signature(aggregator_type.onStart) == inspect.signature(Atom.onStart)
