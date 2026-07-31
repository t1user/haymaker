"""Structured Signal producers, including dataframe calculation models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import singledispatchmethod
from typing import Any, ClassVar

import ib_insync as ibi
import pandas as pd

from ..async_wrappers import QueueShutdownPolicy
from ..base import Atom
from ..datastore import QueuedDataSink
from ..misc import tree
from .messages import Signal, SignalPair, SignalType

log = logging.getLogger(__name__)


def _aware_timestamp(value: Any) -> datetime | None:
    """Convert a dataframe observation label to an aware datetime."""

    if not isinstance(value, (datetime, pd.Timestamp)):
        return None
    timestamp = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


@dataclass(eq=False)
class SignalModel(Atom, ABC):
    """Produce standard Signals for one logical source and Contract.

    Args:
        source_key: Stable identity used by Portfolio, Book, and execution.
        contract: Contract to which generated Signals apply.
        signal_type: State-replacement or event semantics.

    Subclasses implement :meth:`create_signal`. The base :meth:`onData`
    validates that the returned Signal preserves this model's source, resolved
    Contract, and SignalType before emitting it.

    Raises:
        TypeError: If ``source_key`` or ``signal_type`` has the wrong type.
        ValueError: If ``source_key`` is empty.
    """

    source_key: str
    contract: ibi.Contract = field()
    signal_type: SignalType

    output_type: ClassVar[type] = Signal

    def __post_init__(self) -> None:
        """Initialize Atom services and validate Signal identity configuration."""

        Atom.__init__(self)
        if not isinstance(self.source_key, str):
            raise TypeError("source_key must be a string")
        if not self.source_key:
            raise ValueError("source_key must not be empty")
        if not isinstance(self.signal_type, SignalType):
            raise TypeError("signal_type must be a SignalType")

    def onData(self, data: Any, *args: Any) -> None:
        """Create and emit one Signal from arbitrary upstream data."""

        signal = self._validate_signal(self.create_signal(data))
        self.dataEvent.emit(signal)

    def _validate_signal(
        self,
        signal: object,
        *,
        contract: ibi.Contract | None = None,
        producer: str = "create_signal()",
    ) -> Signal:
        """Validate the Signal identity owned by this model."""

        if not isinstance(signal, Signal):
            raise TypeError(f"{producer} must return Signal")
        expected_contract = contract if contract is not None else self.contract
        if expected_contract is None:
            raise RuntimeError("SignalModel Contract is unavailable")
        if signal.source_key != self.source_key:
            raise ValueError(f"{producer} changed source_key")
        if signal.contract != expected_contract:
            raise ValueError(f"{producer} changed Contract")
        if signal.signal_type is not self.signal_type:
            raise ValueError(f"{producer} changed SignalType")
        return signal

    @abstractmethod
    def create_signal(self, data: Any) -> Signal:
        """Return one Signal calculated from upstream data.

        Args:
            data: Arbitrary upstream message accepted by the implementation.

        Returns:
            Signal preserving this model's configured source, resolved Contract,
            and SignalType.
        """


RowToSignal = Callable[[pd.Series, ibi.Contract], Signal]
SignalFields = str | tuple[str, str]


def _validate_signal_fields(value: SignalFields) -> SignalFields:
    """Validate a scalar field name or an entry/exit field pair."""

    if isinstance(value, str):
        if not value:
            raise ValueError("signal_fields must not be empty")
        return value
    if not isinstance(value, tuple):
        raise TypeError("signal_fields must be a field name or a two-field tuple")
    if len(value) != 2:
        raise ValueError("signal_fields tuple must contain exactly two fields")
    if not all(isinstance(field, str) for field in value):
        raise TypeError("signal_fields tuple members must be strings")
    if not all(value):
        raise ValueError("signal_fields tuple members must not be empty")
    if value[0] == value[1]:
        raise ValueError("signal_fields tuple members must be distinct")
    return value


@dataclass(eq=False)
class PandasSignalModel(SignalModel, ABC):
    """Calculate a Signal from the last row of a pandas dataframe.

    Args:
        source_key: Stable logical input identity.
        contract: Signal Contract blueprint.
        signal_type: State or event semantics.
        signal_fields: Calculated row field containing a scalar Signal value,
            or an ``(entry, exit)`` field-name tuple that creates a SignalPair.
        row_to_signal: Optional hook replacing standard row conversion.
        audit_sink: Optional ordered ``DRAIN`` sink for complete calculation
            audit history. Other queued shutdown policies are rejected.

    ``df(data)`` receives a dataframe converted from DataFrame, BarDataList,
    mapping, or dataframe-compatible data. The default row conversion derives
    ``as_of`` from the last row's index and places other row fields in
    metadata. Row order is authoritative: subclasses own sorting, duplicate
    handling, and calculation correctness.
    """

    signal_fields: SignalFields = field(default="signal", kw_only=True)
    row_to_signal: RowToSignal | None = field(default=None, kw_only=True, repr=False)
    audit_sink: QueuedDataSink | None = field(default=None, kw_only=True, repr=False)
    _audit_symbol: str | None = field(default=None, init=False, repr=False)
    _audit_active_con_id: int | None = field(default=None, init=False, repr=False)
    _audit_last_index: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate dataframe configuration and initialize Atom services."""

        if self.audit_sink is not None and (
            self.audit_sink.shutdown_policy is not QueueShutdownPolicy.DRAIN
        ):
            raise ValueError("SignalModel audit_sink must use DRAIN shutdown")
        self.signal_fields = _validate_signal_fields(self.signal_fields)
        super().__post_init__()

    def create_signal(self, data: Any) -> Signal:
        """Calculate, optionally save, and convert the last dataframe row."""

        calculated = self.df(self._as_dataframe(data))
        if not isinstance(calculated, pd.DataFrame):
            raise TypeError("df() must return a pandas.DataFrame")
        if calculated.empty:
            raise ValueError("df() returned an empty dataframe")
        row = calculated.iloc[-1]
        contract = self.contract
        if contract is None:
            raise RuntimeError("SignalModel Contract is unavailable")
        if self.row_to_signal is not None:
            signal = self._validate_row_to_signal(
                self.row_to_signal(row, contract),
                contract,
            )
        else:
            signal = self._default_row_to_signal(
                row,
                source_key=self.source_key,
                contract=contract,
                signal_type=self.signal_type,
                signal_fields=self.signal_fields,
            )
            self._validate_signal(signal, contract=contract)

        audit_reference = self.save_df(calculated)
        if audit_reference is not None:
            signal = replace(
                signal,
                metadata={
                    **signal.metadata,
                    "audit_symbol": audit_reference,
                },
            )
        return signal

    def _validate_row_to_signal(
        self,
        signal: object,
        contract: ibi.Contract,
    ) -> Signal:
        """Validate a Signal returned by the custom row conversion hook."""

        if not isinstance(signal, Signal):
            raise TypeError("row_to_signal must return Signal")
        return self._validate_signal(
            signal,
            contract=contract,
            producer="row_to_signal",
        )

    @staticmethod
    def _default_row_to_signal(
        row: pd.Series,
        *,
        source_key: str,
        contract: ibi.Contract,
        signal_type: SignalType,
        signal_fields: SignalFields,
    ) -> Signal:
        """Convert one calculated row using the standard field mapping."""

        fields = (signal_fields,) if isinstance(signal_fields, str) else signal_fields
        missing = [field for field in fields if field not in row]
        if missing:
            raise KeyError(
                "Calculated row is missing signal field(s): "
                + ", ".join(repr(field) for field in missing)
            )
        metadata = {str(key): value for key, value in row.items() if key not in fields}
        value: float | SignalPair
        if isinstance(signal_fields, str):
            value = row[signal_fields]
        else:
            entry_field, exit_field = signal_fields
            value = SignalPair(
                entry=row[entry_field],
                exit=row[exit_field],
            )
        return Signal(
            source_key=source_key,
            contract=contract,
            value=value,
            signal_type=signal_type,
            as_of=_aware_timestamp(row.name),
            metadata=metadata,
        )

    @singledispatchmethod
    def _as_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert dataframe-compatible upstream data."""

        frame = pd.DataFrame(data)
        if "date" in frame:
            frame = frame.set_index("date")
        return frame

    @_as_dataframe.register
    def _(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    @_as_dataframe.register
    def _(self, data: ibi.BarDataList) -> pd.DataFrame:
        frame = ibi.util.df(data)
        if frame is None:
            return pd.DataFrame()
        if "date" in frame:
            frame = frame.set_index("date")
        return frame

    @abstractmethod
    def df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the complete calculated dataframe used for the Signal.

        Args:
            data: Dataframe converted from the upstream message.

        Returns:
            Calculated dataframe whose last row will produce the Signal.
        """

    def save_df(self, frame: pd.DataFrame) -> str | None:
        """Queue calculated dataframe rows when a save sink is configured.

        Args:
            frame: Complete calculated dataframe returned by :meth:`df`. The
                caller must not mutate it after this method queues it.

        Returns:
            Physical dataframe symbol, or ``None`` when saving is disabled.

        Raises:
            RuntimeError: If saving is enabled before contract selection has
                been initialized.
        """

        if self.audit_sink is None:
            return None
        selector = self.contract_selector
        if selector is None:
            raise RuntimeError(
                "PandasSignalModel cannot save data before its Contract selector "
                "is initialized"
            )
        active = selector.active_contract
        run_started_at = self.runtime.run_started_at
        symbol = (
            f"{self.source_key}_"
            f"{active.localSymbol or active.symbol}_"
            f"{run_started_at.isoformat()}"
        )
        is_new_generation = (
            self._audit_active_con_id != active.conId or self._audit_symbol != symbol
        )
        metadata = {
            "source_key": self.source_key,
            "run_started_at": run_started_at,
            "active_contract": tree(active),
        }
        if is_new_generation:
            self.audit_sink.enqueue_write(symbol, frame, metadata)
            self._audit_symbol = symbol
            self._audit_active_con_id = active.conId
            log.info("Started Signal dataframe generation %s", symbol)
        else:
            new_rows = frame
            if self._audit_last_index is not None:
                new_rows = frame.loc[frame.index > self._audit_last_index]
            if not new_rows.empty:
                self.audit_sink.enqueue_append(symbol, new_rows, metadata)
                log.debug("Appended Signal dataframe generation %s", symbol)
        self._audit_last_index = frame.index[-1]
        return symbol


__all__ = ["PandasSignalModel", "SignalModel"]
