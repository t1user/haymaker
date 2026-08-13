"""Structured Signal producers, including dataframe calculation models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import singledispatchmethod
from typing import Any, ClassVar, final

import ib_insync as ibi
import pandas as pd

from ..base import Atom
from ..datastore import SignalFramePersistence
from ..validators import non_empty_string
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
    """Keep the calculated frame available for optional emission-time saving."""

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
        self.source_key = non_empty_string(self.source_key, "source_key")
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
        directly when a calculated Signal is needed without emitting it or
        invoking emission-time persistence; subclasses customize
        :meth:`calculate_signal` and
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
        return self._signal_from_calculation(calculation)

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
        persistence: ``False`` disables calculation-data persistence. ``True``
            creates a model-owned persistence object from runtime YAML defaults.

    ``df(data)`` receives a dataframe converted from DataFrame, BarDataList,
    mapping, or dataframe-compatible data. The default row conversion derives
    ``as_of`` from the last row's index and places other row fields in
    metadata. Override :meth:`row_to_calculation` when field selection cannot
    express the required conversion. Row order is authoritative: subclasses
    own sorting, duplicate handling, and calculation correctness.
    """

    signal_fields: SignalFields = field(default="signal", kw_only=True)
    metadata_fields: Collection[str] | None = field(default=None, kw_only=True)
    persistence: bool | SignalFramePersistence = field(
        default=False,
        kw_only=True,
        repr=False,
    )

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

        self.signal_fields = self._validate_signal_fields(self.signal_fields)
        self.metadata_fields = self._normalize_metadata_fields(self.metadata_fields)
        super().__post_init__()
        self.persistence = self._resolve_persistence(self.persistence)

    @staticmethod
    def _validate_persistence(
        persistence: object,
    ) -> SignalFramePersistence:
        """Return a structurally valid custom persistence object."""

        if not isinstance(persistence, SignalFramePersistence):
            raise TypeError(
                "persistence must be True, False, or a SignalFramePersistence"
            )
        return persistence

    def _resolve_persistence(
        self,
        persistence: bool | SignalFramePersistence,
    ) -> bool | SignalFramePersistence:
        """Resolve the runtime default once while preserving explicit disablement."""

        if persistence is False:
            return False
        if persistence is True:
            persistence = self.runtime.signal_persistence_factory()
        return self._validate_persistence(persistence)

    @final
    def onData(self, data: Any, *args: Any) -> None:
        """Calculate, queue optional persistence, and emit one Signal.

        Persistence only queues work and never waits for storage I/O. A failure
        to accept persistence work is logged and the Signal is still emitted,
        without an ``audit_symbol`` reference.

        Args:
            data: Dataframe-compatible upstream message.
            *args: Additional event arguments, accepted for eventkit callbacks
                and otherwise ignored.

        Emits:
            Signal: Validated Signal, with ``audit_symbol`` metadata only after
            persistence queue acceptance.
        """

        calculation = self.calculate_signal(data)
        signal = self._signal_from_calculation(calculation)
        if not isinstance(calculation, _PandasSignalCalculation):
            raise TypeError("PandasSignalModel calculation is missing its dataframe")
        signal = self._with_persistence_reference(signal, calculation.frame)
        self.dataEvent.emit(signal)

    def _with_persistence_reference(
        self,
        signal: Signal,
        frame: pd.DataFrame,
    ) -> Signal:
        """Queue calculated data and add its accepted storage reference."""

        persistence = self.persistence
        if persistence is False:
            return signal
        if persistence is True:
            raise RuntimeError("Signal persistence was not initialized")
        try:
            reference = persistence.save(
                frame,
                source_key=self.source_key,
                active_contract=self.contract_selector.active_contract,
                run_started_at=self.runtime.run_started_at,
            )
            if reference is not None and not isinstance(reference, str):
                raise TypeError("Signal persistence reference must be a string or None")
        except Exception:
            log.exception(
                "%s could not queue calculated dataframe persistence; "
                "emitting Signal without an audit reference",
                self,
            )
            return signal
        if reference is None:
            return signal
        return replace(
            signal,
            metadata={**signal.metadata, "audit_symbol": reference},
        )

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
        calculation = self.row_to_calculation(row)
        if not isinstance(calculation, SignalCalculation):
            raise TypeError("row_to_calculation must return SignalCalculation")

        return _PandasSignalCalculation(
            value=calculation.value,
            metadata=calculation.metadata,
            as_of=calculation.as_of,
            frame=calculated,
        )

    def row_to_calculation(
        self,
        row: pd.Series,
    ) -> SignalCalculation:
        """Convert the final calculated row into Signal contents.

        The default implementation reads the configured ``signal_fields`` and
        ``metadata_fields``. Override this method when a row needs custom value,
        metadata, or observation-time conversion; return only calculated fields
        and leave Signal identity to the framework.

        Args:
            row: Final row of the dataframe returned by :meth:`df`.

        Returns:
            Calculated Signal value, metadata, and observation time.

        Raises:
            KeyError: If a configured signal or metadata field is absent.
        """

        signal_fields = self.signal_fields
        metadata_fields = self.metadata_fields
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


__all__ = ["PandasSignalModel", "SignalCalculation", "SignalModel"]
