from dataclasses import is_dataclass

import pytest

from haymaker.base import Atom
from haymaker.components import (
    DfAggregator,
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
        DfAggregator,
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
