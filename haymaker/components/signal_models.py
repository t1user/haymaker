"""Structured Signal producers, including dataframe calculation models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import singledispatchmethod
from typing import Any, ClassVar, final

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


@dataclass(frozen=True, kw_only=True)
class SignalCalculation:
    """Carry user-calculated fields from which SignalModel builds a Signal.

    Use this as the single return type of :meth:`SignalModel.calculate_signal`.
    SignalModel supplies source identity, the resolved Contract, SignalType, and
    ``created_at`` when it constructs the emitted :class:`Signal`.

    Args:
        value: Finite scalar value or one-to-one entry/exit pair.
        metadata: Optional calculated fields consumed downstream. Omit it when
            the calculation has no metadata.
        as_of: Effective observation time. For left-labelled IB bars this is
            the time at which the bar interval started, not when calculation
            completed.

    Note:
        Intrinsic value, metadata, and timestamp validation occur when the
        framework constructs the immutable Signal.
    """

    value: float | SignalPair
    metadata: Mapping[str, Any] = field(default_factory=dict)
    as_of: datetime | None = None


@dataclass(frozen=True, kw_only=True)
class _PandasSignalCalculation(SignalCalculation):
    """Keep the calculated frame alive until its validated Signal is saved."""

    frame: pd.DataFrame = field(repr=False, compare=False)


@dataclass(eq=False)
class SignalModel(Atom, ABC):
    """Produce standard Signals for one logical source and Contract.

    Args:
        source_key: Stable identity used by Portfolio, Book, and execution.
        contract: Contract to which generated Signals apply.
        signal_type: State-replacement or event semantics.

    Subclasses implement :meth:`calculate_signal` and return only calculated
    value, metadata, and observation time. SignalModel owns construction of the
    immutable Signal envelope and emits it from :meth:`onData`.

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
        """Create and emit one Signal from arbitrary upstream data.

        Args:
            data: Message supplied by the connected upstream Atom.
            *args: Additional event arguments, accepted for eventkit callbacks
                and otherwise ignored.

        Emits:
            Signal: The validated Signal returned by :meth:`create_signal`.
        """

        signal = self.create_signal(data)
        self.dataEvent.emit(signal)

    @final
    def create_signal(self, data: Any) -> Signal:
        """Calculate and return one framework-owned Signal.

        This method is the template boundary used by :meth:`onData`. Call it
        directly when a calculated Signal is needed without emitting it;
        subclasses customize :meth:`calculate_signal` and
        :meth:`validate_signal_value` instead.

        Args:
            data: Arbitrary upstream message accepted by the implementation.

        Returns:
            Immutable Signal with framework-owned identity and creation time.

        Raises:
            TypeError: If the calculation has the wrong result type or the
                resulting Signal is structurally invalid.
            ValueError: If intrinsic or model-specific Signal validation fails.
        """

        calculation = self.calculate_signal(data)
        signal = self._signal_from_calculation(calculation)
        additional_metadata = self._additional_signal_metadata(calculation)
        if additional_metadata:
            signal = replace(
                signal,
                metadata={**signal.metadata, **additional_metadata},
            )
        return signal

    def _signal_from_calculation(self, calculation: object) -> Signal:
        """Construct and validate the Signal envelope owned by this model."""

        if not isinstance(calculation, SignalCalculation):
            raise TypeError("calculate_signal() must return SignalCalculation")
        contract = self.contract
        signal = Signal(
            source_key=self.source_key,
            contract=contract,
            value=calculation.value,
            signal_type=self.signal_type,
            as_of=calculation.as_of,
            metadata=calculation.metadata,
        )
        self.validate_signal_value(signal.value)
        return signal

    def _additional_signal_metadata(
        self,
        calculation: SignalCalculation,
    ) -> Mapping[str, Any]:
        """Return framework metadata to add after Signal validation."""

        return {}

    def validate_signal_value(self, value: float | SignalPair) -> None:
        """Validate model-specific value constraints before Signal emission.

        Override this hook to reject values that are structurally valid Signals
        but invalid for a particular model, such as non-binary scalar values.
        The base implementation accepts every value supported by Signal.

        Args:
            value: Signal-normalized finite float or SignalPair.

        Raises:
            ValueError: When an override rejects the calculated value.
        """

    @abstractmethod
    def calculate_signal(self, data: Any) -> SignalCalculation:
        """Calculate the user-owned contents of one Signal.

        Args:
            data: Arbitrary upstream message accepted by the implementation.

        Returns:
            Value, optional metadata, and optional effective observation time.
        """


RowToCalculation = Callable[[pd.Series], SignalCalculation]
SignalFields = str | tuple[str, str]


@dataclass(eq=False)
class PandasSignalModel(SignalModel, ABC):
    """Calculate a Signal from the last row of a pandas dataframe.

    Args:
        source_key: Stable logical input identity.
        contract: Signal Contract blueprint.
        signal_type: State or event semantics.
        signal_fields: Calculated row field containing a scalar Signal value,
            or an ``(entry, exit)`` field-name tuple that creates a SignalPair.
        metadata_fields: Row fields copied into Signal metadata. ``None`` copies
            every non-signal field, an empty collection copies none, and an
            explicit collection copies only those fields.
        row_to_calculation: Optional hook replacing standard row conversion. It
            returns SignalCalculation and never supplies Signal identity.
        audit_sink: Optional ordered ``DRAIN`` sink for complete calculation
            audit history. Other queued shutdown policies are rejected.

    ``df(data)`` receives a dataframe converted from DataFrame, BarDataList,
    mapping, or dataframe-compatible data. The default row conversion derives
    ``as_of`` from the last row's index and places other row fields in
    metadata. Row order is authoritative: subclasses own sorting, duplicate
    handling, and calculation correctness.
    """

    signal_fields: SignalFields = field(default="signal", kw_only=True)
    metadata_fields: Collection[str] | None = field(default=None, kw_only=True)
    row_to_calculation: RowToCalculation | None = field(
        default=None, kw_only=True, repr=False
    )
    audit_sink: QueuedDataSink | None = field(default=None, kw_only=True, repr=False)
    _audit_symbol: str | None = field(default=None, init=False, repr=False)
    _audit_active_con_id: int | None = field(default=None, init=False, repr=False)
    _audit_last_index: Any = field(default=None, init=False, repr=False)

    @staticmethod
    def _validate_signal_fields(value: SignalFields) -> SignalFields:
        """Validate the shape of a scalar field name or entry/exit field pair."""

        if isinstance(value, str):
            return value
        if not isinstance(value, tuple):
            raise TypeError("signal_fields must be a field name or a two-field tuple")
        if len(value) != 2:
            raise ValueError("signal_fields tuple must contain exactly two fields")
        if not all(isinstance(field, str) for field in value):
            raise TypeError("signal_fields tuple members must be strings")
        return value

    @staticmethod
    def _normalize_metadata_fields(
        value: Collection[str] | None,
    ) -> tuple[str, ...] | None:
        """Normalize an optional collection of dataframe metadata columns."""

        if value is None:
            return None
        if isinstance(value, str) or not isinstance(value, Collection):
            raise TypeError(
                "metadata_fields must be a collection of field names or None"
            )
        if not all(isinstance(field, str) for field in value):
            raise TypeError("metadata_fields members must be strings")
        return tuple(value)

    def __post_init__(self) -> None:
        """Validate dataframe configuration and initialize Atom services."""

        if self.audit_sink is not None and (
            self.audit_sink.shutdown_policy is not QueueShutdownPolicy.DRAIN
        ):
            raise ValueError("SignalModel audit_sink must use DRAIN shutdown")
        self.signal_fields = self._validate_signal_fields(self.signal_fields)
        self.metadata_fields = self._normalize_metadata_fields(self.metadata_fields)
        super().__post_init__()

    @final
    def calculate_signal(self, data: Any) -> SignalCalculation:
        """Calculate Signal contents from the last dataframe row.

        Args:
            data: DataFrame, BarDataList, mapping, or dataframe-compatible
                upstream message.

        Returns:
            Calculation result converted from the final row returned by
            :meth:`df`.

        Raises:
            TypeError: If :meth:`df` or ``row_to_calculation`` returns the wrong
                type.
            ValueError: If :meth:`df` returns an empty dataframe.
            KeyError: If a configured signal or metadata field is absent.
        """

        calculated = self.df(self._as_dataframe(data))
        if not isinstance(calculated, pd.DataFrame):
            raise TypeError("df() must return a pandas.DataFrame")
        if calculated.empty:
            raise ValueError("df() returned an empty dataframe")
        row = calculated.iloc[-1]
        if self.row_to_calculation is not None:
            calculation = self.row_to_calculation(row)
            if not isinstance(calculation, SignalCalculation):
                raise TypeError("row_to_calculation must return SignalCalculation")
        else:
            calculation = self._default_row_to_calculation(
                row,
                signal_fields=self.signal_fields,
                metadata_fields=self.metadata_fields,
            )
        return _PandasSignalCalculation(
            value=calculation.value,
            metadata=calculation.metadata,
            as_of=calculation.as_of,
            frame=calculated,
        )

    def _additional_signal_metadata(
        self,
        calculation: SignalCalculation,
    ) -> Mapping[str, Any]:
        """Save a validated dataframe and return its optional audit reference."""

        if not isinstance(calculation, _PandasSignalCalculation):
            raise TypeError("PandasSignalModel calculation is missing its dataframe")
        audit_reference = self.save_df(calculation.frame)
        if audit_reference is not None:
            return {"audit_symbol": audit_reference}
        return {}

    @staticmethod
    def _default_row_to_calculation(
        row: pd.Series,
        *,
        signal_fields: SignalFields,
        metadata_fields: Collection[str] | None,
    ) -> SignalCalculation:
        """Convert one calculated row using standard value and metadata fields."""

        fields = (signal_fields,) if isinstance(signal_fields, str) else signal_fields
        missing = [field for field in fields if field not in row]
        if missing:
            raise KeyError(
                "Calculated row is missing signal field(s): "
                + ", ".join(repr(field) for field in missing)
            )
        if metadata_fields is None:
            metadata = {
                str(key): value for key, value in row.items() if key not in fields
            }
        else:
            missing_metadata = [field for field in metadata_fields if field not in row]
            if missing_metadata:
                raise KeyError(
                    "Calculated row is missing metadata field(s): "
                    + ", ".join(repr(field) for field in missing_metadata)
                )
            metadata = {field: row[field] for field in metadata_fields}
        value: float | SignalPair
        if isinstance(signal_fields, str):
            value = row[signal_fields]
        else:
            entry_field, exit_field = signal_fields
            value = SignalPair(
                entry=row[entry_field],
                exit=row[exit_field],
            )
        return SignalCalculation(
            value=value,
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


__all__ = ["PandasSignalModel", "SignalCalculation", "SignalModel"]
