"""Structured Signal producers, including dataframe calculation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from functools import singledispatchmethod
import logging
from typing import Any, ClassVar

import ib_insync as ibi
import pandas as pd

from ..async_wrappers import QueueShutdownPolicy
from ..base import Atom
from ..datastore import AsyncDataStore, QueuedDataSink
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


class SignalModel(Atom, ABC):
    """Produce standard Signals for one logical source and Contract.

    Args:
        source_key: Stable identity used by Portfolio, Book, and execution.
        contract: Contract to which generated Signals apply.
        signal_type: State-replacement or event semantics.
        auto_roll_futures: Whether this source participates in Controller-owned
            automatic futures rolling.

    Subclasses implement ``create_signal``; ``onData`` emits its
    returned immutable Signal.
    """

    output_type: ClassVar[type] = Signal

    def __init__(
        self,
        source_key: str,
        contract: ibi.Contract,
        signal_type: SignalType,
        *,
        auto_roll_futures: bool = True,
    ) -> None:
        super().__init__()
        if not source_key:
            raise ValueError("source_key must not be empty")
        if not isinstance(signal_type, SignalType):
            raise TypeError("signal_type must be a SignalType")
        self.source_key = source_key
        self.contract = contract
        self.signal_type = signal_type
        self.auto_roll_futures = auto_roll_futures
        self._register_future_roll_policy()

    def _register_future_roll_policy(self) -> None:
        policies = self.runtime.future_roll_policies
        previous = policies.get(self.source_key)
        if previous is not None and previous != self.auto_roll_futures:
            raise ValueError(
                "Conflicting auto_roll_futures values for source "
                f"{self.source_key!r}"
            )
        policies[self.source_key] = self.auto_roll_futures

    def onData(self, data: Any, *args: Any) -> None:
        """Create and emit one Signal from arbitrary upstream data."""

        signal = self.create_signal(data)
        if not isinstance(signal, Signal):
            raise TypeError("create_signal() must return Signal")
        self.dataEvent.emit(signal)

    @abstractmethod
    def create_signal(self, data: Any) -> Signal:
        """Return one Signal calculated from upstream data."""


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


class PandasSignalModel(SignalModel, ABC):
    """Calculate a Signal from the latest row of a pandas dataframe.

    Args:
        source_key: Stable logical input identity.
        contract: Signal Contract blueprint.
        signal_type: State or event semantics.
        signal_fields: Calculated row field containing a scalar Signal value,
            or an ``(entry, exit)`` field-name tuple that creates a SignalPair.
        row_to_signal: Optional hook replacing standard row conversion.
        audit_sink: Optional ordered ``DRAIN`` sink for complete calculation
            audit history. Other queued shutdown policies are rejected.
        auto_roll_futures: Controller-owned futures-roll policy.

    ``df(data)`` receives a dataframe converted from DataFrame, BarDataList,
    mapping, or dataframe-compatible data. The default row conversion derives
    ``as_of`` from the latest index and places other row fields in metadata.
    """

    def __init__(
        self,
        source_key: str,
        contract: ibi.Contract,
        signal_type: SignalType,
        *,
        signal_fields: SignalFields = "signal",
        row_to_signal: RowToSignal | None = None,
        audit_sink: QueuedDataSink | None = None,
        auto_roll_futures: bool = True,
    ) -> None:
        if audit_sink is not None and (
            audit_sink.shutdown_policy is not QueueShutdownPolicy.DRAIN
        ):
            raise ValueError("SignalModel audit_sink must use DRAIN shutdown")
        self.signal_fields = _validate_signal_fields(signal_fields)
        self.row_to_signal = row_to_signal
        self.audit_sink = audit_sink
        self._audit_symbol: str | None = None
        self._audit_active_con_id: int | None = None
        self._audit_last_index: Any = None
        super().__init__(
            source_key,
            contract,
            signal_type,
            auto_roll_futures=auto_roll_futures,
        )

    def create_signal(self, data: Any) -> Signal:
        """Calculate, audit, and convert the latest dataframe row."""

        calculated = self.df(self._as_dataframe(data))
        if not isinstance(calculated, pd.DataFrame):
            raise TypeError("df() must return a pandas.DataFrame")
        if calculated.empty:
            raise ValueError("df() returned an empty dataframe")
        row = calculated.iloc[-1]
        contract = self._signal_contract()
        if self.row_to_signal is not None:
            signal = self.row_to_signal(row, contract)
            if not isinstance(signal, Signal):
                raise TypeError("row_to_signal must return Signal")
            if signal.source_key != self.source_key:
                raise ValueError("row_to_signal changed source_key")
            if signal.contract != contract:
                raise ValueError("row_to_signal changed Contract")
            if signal.signal_type is not self.signal_type:
                raise ValueError("row_to_signal changed SignalType")
        else:
            fields = (
                (self.signal_fields,)
                if isinstance(self.signal_fields, str)
                else self.signal_fields
            )
            missing = [field for field in fields if field not in row]
            if missing:
                raise KeyError(
                    "Calculated row is missing signal field(s): "
                    + ", ".join(repr(field) for field in missing)
                )
            metadata = {
                str(key): value for key, value in row.items() if key not in fields
            }
            value: float | SignalPair
            if isinstance(self.signal_fields, str):
                value = row[self.signal_fields]
            else:
                entry_field, exit_field = self.signal_fields
                value = SignalPair(
                    entry=row[entry_field],
                    exit=row[exit_field],
                )
            signal = Signal(
                source_key=self.source_key,
                contract=contract,
                value=value,
                signal_type=self.signal_type,
                as_of=_aware_timestamp(row.name),
                metadata=metadata,
            )

        audit_reference = self._persist_audit(calculated)
        if audit_reference is not None:
            signal = replace(
                signal,
                metadata={
                    **signal.metadata,
                    "audit_symbol": audit_reference,
                },
            )
        return signal

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
        """Return the complete calculated dataframe used for the Signal."""

    def _active_contract(self) -> ibi.Contract:
        """Return ACTIVE for audit identity, independent of transaction role."""

        try:
            selector = self.contract_selector
        except KeyError:
            selector = None
        if selector is not None:
            return selector.active_contract
        return self._signal_contract()

    def _signal_contract(self) -> ibi.Contract:
        """Return the configured Contract after runtime registry resolution."""

        contract = self.contract
        if contract is None:
            raise RuntimeError("SignalModel Contract is unavailable")
        return contract

    def _persist_audit(self, frame: pd.DataFrame) -> str | None:
        """Queue ordered audit writes after a successful calculation."""

        if self.audit_sink is None:
            return None
        active = self._active_contract()
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
            "active_contract": ibi.util.tree(active),
            "observation_start": frame.index[0],
            "observation_end": frame.index[-1],
            "status": "complete",
        }
        if is_new_generation:
            self.audit_sink.enqueue_write(symbol, frame, metadata)
            self._audit_symbol = symbol
            self._audit_active_con_id = active.conId
            log.info("Started Signal audit generation %s", symbol)
        else:
            new_rows = frame
            if self._audit_last_index is not None:
                new_rows = frame.loc[frame.index > self._audit_last_index]
            if not new_rows.empty:
                self.audit_sink.enqueue_append(symbol, new_rows, metadata)
                log.debug("Appended Signal audit generation %s", symbol)
        self._audit_last_index = frame.index[-1]
        return symbol


async def read_signal_audit(
    store: AsyncDataStore,
    *,
    source_key: str,
    created_at: datetime,
    as_of: datetime | None = None,
) -> pd.DataFrame | None:
    """Read the latest audit run created no later than a Signal.

    Args:
        store: Awaited dataframe store used for the audit library.
        source_key: Signal source identity.
        created_at: Signal creation time used to bound candidate runs.
        as_of: Optional final observation timestamp.

    Returns:
        Audit dataframe through ``as_of``, or ``None`` when no matching run
        exists.
    """

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    if as_of is not None and (as_of.tzinfo is None or as_of.utcoffset() is None):
        raise ValueError("as_of must be timezone-aware")
    prefix = f"{source_key}_"
    candidates: list[tuple[datetime, str]] = []
    for symbol in await store.keys():
        if not symbol.startswith(prefix):
            continue
        metadata: Mapping[str, Any] = await store.read_metadata(symbol)
        if metadata.get("source_key") != source_key:
            continue
        run_started_at = metadata.get("run_started_at")
        if isinstance(run_started_at, str):
            run_started_at = datetime.fromisoformat(run_started_at)
        if isinstance(run_started_at, datetime) and (
            run_started_at.tzinfo is None or run_started_at.utcoffset() is None
        ):
            run_started_at = run_started_at.replace(tzinfo=timezone.utc)
        if isinstance(run_started_at, datetime) and run_started_at <= created_at:
            candidates.append((run_started_at, symbol))
    if not candidates:
        return None
    _, symbol = max(candidates)
    return await store.read(symbol, end_date=as_of)


__all__ = ["PandasSignalModel", "SignalModel", "read_signal_audit"]
