"""Read historical bars and contract metadata for replay simulations.

This module deliberately depends on a small, read-only datastore protocol.  It
turns the collections written by :mod:`haymaker.dataloader` into qualified
``ib_insync`` contracts and cumulative ``BarDataList`` snapshots without
contacting Interactive Brokers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable

import ib_insync as ibi
import pandas as pd

ReplayTimestamp: TypeAlias = date | datetime

_CONTRACT_FIELD_NAMES = tuple(field.name for field in fields(ibi.Contract))
_CONTRACT_FIELDS = frozenset(_CONTRACT_FIELD_NAMES)
_CONTRACT_DETAILS_FIELDS = frozenset(
    field.name for field in fields(ibi.ContractDetails)
)
_REQUIRED_BAR_COLUMNS = frozenset(
    {"open", "high", "low", "close", "volume", "average", "barCount"}
)
_REQUIRED_PRICE_COLUMNS = frozenset({"open", "high", "low", "close"})
_DETAIL_ALIASES = {"min_tick": "minTick", "name": "longName"}
_NON_IDENTITY_CONTRACT_FIELDS = frozenset({"conId", "secType", "includeExpired"})
_CONT_FUTURE_IDENTITY_FIELDS = (
    "symbol",
    "multiplier",
    "exchange",
    "primaryExchange",
    "currency",
    "tradingClass",
    "secIdType",
    "secId",
    "issuerId",
)
_DEFAULT_CONTRACT = ibi.Contract()


class BacktestDataError(RuntimeError):
    """Base exception for replay-data loading failures."""


class BacktestDataNotLoadedError(BacktestDataError):
    """Raised when loaded data is accessed before :meth:`load`."""


class InvalidBacktestDataError(BacktestDataError, ValueError):
    """Raised when persisted bars or metadata cannot be replayed safely."""


class MissingBacktestDataError(BacktestDataError, LookupError):
    """Raised when no loaded series matches a requested contract."""


class AmbiguousBacktestDataError(BacktestDataError, LookupError):
    """Raised when a contract blueprint matches multiple loaded series."""


class MissingContractMetadataError(BacktestDataError, KeyError):
    """Raised when persisted metadata lacks a required contract detail."""


@runtime_checkable
class BacktestDataStore(Protocol):
    """Read-only async interface required by :class:`BacktestDataRepository`.

    Datastore keys are passed back as strings.  This is important for dataloader
    collections because an unqualified contract does not necessarily reproduce
    the collection name generated from its qualified ``localSymbol``.
    """

    async def keys(self) -> list[str]:
        """Return collection names available in the configured library."""

        ...

    async def read(
        self,
        symbol: str,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame | None:
        """Read one dataframe by its persisted collection name."""

        ...

    async def read_metadata(self, symbol: str) -> dict[str, Any]:
        """Read top-level metadata for one persisted collection."""

        ...


@dataclass(frozen=True)
class StoredSeries:
    """One validated historical series and its reconstructed contract.

    Args:
        key: Persisted datastore collection name.
        contract: Contract rebuilt from top-level datastore metadata.
        metadata: Read-only copy of all metadata, including opaque extra fields.
        frame: Complete validated dataframe.  Rows before the replay start are
            intentionally retained so framework components can warm up.
    """

    key: str
    contract: ibi.Contract
    metadata: Mapping[str, Any]
    frame: pd.DataFrame

    @property
    def contract_details(self) -> ibi.ContractDetails:
        """Return broker-style details built entirely from persisted metadata."""

        values = _contract_details_values(self.metadata)
        return ibi.ContractDetails(contract=self.contract, **values)

    def bars_through(self, timestamp: ReplayTimestamp) -> ibi.BarDataList:
        """Return a cumulative bar snapshot ending at ``timestamp``.

        Args:
            timestamp: Replay clock point.  It must use the same date category as
                this series: a calendar date for daily data or an aware datetime
                for intraday data.

        Returns:
            A new ``BarDataList`` containing every stored row no later than the
            requested timestamp, including warmup rows before replay start.
        """

        point = _normalize_requested_timestamp(timestamp, self.frame.index, self.key)
        frame = self.frame.loc[self.frame.index <= point]
        bars = ibi.BarDataList(
            _row_to_bar(index, row) for index, row in frame.iterrows()
        )
        bars.contract = self.contract
        return bars

    def bar_at(self, timestamp: ReplayTimestamp) -> ibi.BarData | None:
        """Return the stored bar exactly at ``timestamp``, if one exists.

        Args:
            timestamp: Replay point using this series' date category.

        Returns:
            A newly constructed ``BarData`` or ``None`` when the contract had no
            observed session at that replay point.
        """

        point = _normalize_requested_timestamp(timestamp, self.frame.index, self.key)
        try:
            row = self.frame.loc[point]
        except KeyError:
            return None
        if not isinstance(row, pd.Series):
            raise InvalidBacktestDataError(
                f"Backtest collection {self.key!r} has duplicate timestamp {point!r}"
            )
        return _row_to_bar(point, row)


class BacktestDataRepository:
    """Load and index dataloader history for deterministic replay.

    Args:
        store: Read-only async datastore for one dataloader library.
        start: Optional first replay clock point.  Earlier rows remain present in
            each :class:`StoredSeries` as warmup history.
        end: Optional final replay clock point.

    Blueprints are optional.  When registered, only matching datastore series
    are loaded.  An explicit contract primarily matches by ``conId``.  A
    ``ContFuture`` blueprint selects the explicit ``FUT`` chain for the same
    root, exchange, currency, and trading class rather than a persisted
    ``CONTFUT`` series.
    """

    def __init__(
        self,
        store: BacktestDataStore,
        *,
        start: ReplayTimestamp | None = None,
        end: ReplayTimestamp | None = None,
    ) -> None:
        self._store = store
        self._start = _normalize_bound(start, "start")
        self._end = _normalize_bound(end, "end")
        _validate_bound_pair(self._start, self._end)
        self._blueprints: list[ibi.Contract] = []
        self._series: tuple[StoredSeries, ...] | None = None
        self._timestamps: tuple[ReplayTimestamp, ...] | None = None

    def register(self, *blueprints: ibi.Contract) -> None:
        """Register contract blueprints used to filter datastore collections.

        Args:
            *blueprints: Concrete contracts or partial contract specifications.

        Raises:
            TypeError: If any blueprint is not an ``ib_insync.Contract``.
            BacktestDataError: If registration is attempted after loading.
        """

        if self._series is not None:
            raise BacktestDataError("Contract blueprints cannot change after load()")
        for blueprint in blueprints:
            if not isinstance(blueprint, ibi.Contract):
                raise TypeError("Backtest contract blueprints must be Contracts")
            self._blueprints.append(blueprint)

    async def load(self) -> "BacktestDataRepository":
        """Load, validate, and index every matching datastore collection.

        Returns:
            This repository, ready for synchronous replay access.

        Raises:
            InvalidBacktestDataError: If matching persisted data has an invalid
                contract, index, or OHLC dataframe.
            MissingBacktestDataError: If a registered blueprint has no matching
                series.
        """

        loaded: list[StoredSeries] = []
        matched_blueprints: set[int] = set()
        keys = await self._store.keys()
        if any(not isinstance(key, str) for key in keys):
            raise InvalidBacktestDataError("Backtest datastore keys must be strings")
        if len(keys) != len(set(keys)):
            raise InvalidBacktestDataError("Backtest datastore returned duplicate keys")

        for key in sorted(keys):
            raw_metadata = await self._store.read_metadata(key)
            if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
                raise InvalidBacktestDataError(
                    f"Backtest collection {key!r} has non-mapping metadata"
                )
            metadata = dict(raw_metadata or {})
            if self._blueprints and not any(
                _metadata_may_match_blueprint(blueprint, metadata)
                for blueprint in self._blueprints
            ):
                continue
            contract = _contract_from_metadata(key, metadata)
            matches = self._matching_blueprint_indexes(contract)
            if self._blueprints and not matches:
                continue
            frame = await self._store.read(key)
            if frame is None:
                raise InvalidBacktestDataError(
                    f"Backtest collection {key!r} has metadata but no dataframe"
                )
            loaded.append(
                StoredSeries(
                    key=key,
                    contract=contract,
                    metadata=MappingProxyType(metadata.copy()),
                    frame=_validate_frame(key, frame),
                )
            )
            matched_blueprints.update(matches)

        missing = [
            blueprint
            for index, blueprint in enumerate(self._blueprints)
            if index not in matched_blueprints
        ]
        if missing:
            raise MissingBacktestDataError(
                "No persisted series matches contract blueprint(s): "
                + ", ".join(map(str, missing))
            )

        self._series = tuple(sorted(loaded, key=lambda item: item.key))
        self._timestamps = self._build_timestamps(self._series)
        return self

    @property
    def series(self) -> tuple[StoredSeries, ...]:
        """Return loaded series in deterministic collection-name order."""

        if self._series is None:
            raise BacktestDataNotLoadedError("Call load() before accessing series")
        return self._series

    @property
    def timestamps(self) -> tuple[ReplayTimestamp, ...]:
        """Return the sorted union of observed replay timestamps.

        Missing timestamps are not synthesized.  The optional start/end bounds
        apply only to this replay clock, not to the warmup rows retained in each
        series.
        """

        if self._timestamps is None:
            raise BacktestDataNotLoadedError("Call load() before accessing timestamps")
        return self._timestamps

    def series_for(self, contract: ibi.Contract) -> StoredSeries:
        """Return the unique loaded series for a concrete contract.

        Args:
            contract: Usually a contract supplied by a framework component.

        Raises:
            MissingBacktestDataError: If no loaded series matches.
            AmbiguousBacktestDataError: If a partial contract matches more than
                one series, such as a futures root without a ``conId``.
        """

        matches = [
            series
            for series in self.series
            if _lookup_matches(contract, series.contract)
        ]
        if not matches:
            raise MissingBacktestDataError(f"No loaded series matches {contract}")
        if len(matches) > 1:
            raise AmbiguousBacktestDataError(
                f"Contract {contract} matches multiple persisted series: "
                + ", ".join(series.key for series in matches)
            )
        return matches[0]

    def metadata_for(self, contract: ibi.Contract) -> Mapping[str, Any]:
        """Return all persisted metadata for a concrete contract.

        Exact lookup uses ``conId`` when the requested contract has one.  This
        allows broker adapters to obtain commission, multiplier, and tick-size
        data without reaching into repository internals.
        """

        return self.series_for(contract).metadata

    def contract_details(self, contract: ibi.Contract) -> ibi.ContractDetails:
        """Return ``ContractDetails`` derived only from persisted metadata."""

        return self.series_for(contract).contract_details

    def details_for_blueprint(
        self, blueprint: ibi.Contract
    ) -> list[ibi.ContractDetails]:
        """Return all stored contract details matching a blueprint.

        Args:
            blueprint: A concrete or partial contract specification.  A
                ``ContFuture`` selects its explicit ``FUT`` chain.

        Returns:
            Newly constructed ``ContractDetails`` values ordered by exact
            contract expiry, then collection key.

        Raises:
            MissingBacktestDataError: If the blueprint matches no loaded series.
        """

        matches = [
            series
            for series in self.series
            if _blueprint_matches(blueprint, series.contract)
        ]
        if not matches:
            raise MissingBacktestDataError(
                f"No loaded series matches contract blueprint {blueprint}"
            )
        matches.sort(key=_series_expiry_key)
        return [series.contract_details for series in matches]

    def require_metadata(self, contract: ibi.Contract, field: str) -> Any:
        """Return a required metadata value, accepting detail-name aliases.

        Args:
            contract: Concrete contract identifying one loaded series.
            field: Metadata field, for example ``commission`` or ``minTick``.

        Raises:
            MissingContractMetadataError: If the field is absent.
        """

        metadata = self.metadata_for(contract)
        for candidate in _metadata_candidates(field):
            if candidate in metadata and metadata[candidate] is not None:
                return metadata[candidate]
        raise MissingContractMetadataError(
            f"Contract {contract} has no persisted metadata for {field!r}"
        )

    def bars_through(
        self, contract: ibi.Contract, timestamp: ReplayTimestamp
    ) -> ibi.BarDataList:
        """Return the contract's cumulative history through a replay point."""

        return self.series_for(contract).bars_through(timestamp)

    def bar_at(
        self, contract: ibi.Contract, timestamp: ReplayTimestamp
    ) -> ibi.BarData | None:
        """Return one contract's bar exactly at a replay point, if present."""

        return self.series_for(contract).bar_at(timestamp)

    def series_at(self, timestamp: ReplayTimestamp) -> tuple[StoredSeries, ...]:
        """Return series that contain an observed row at ``timestamp``.

        This implements the backtester's session rule: absence of a stored bar
        means there was no session for that contract at that replay point.
        """

        matches: list[StoredSeries] = []
        for series in self.series:
            point = _normalize_requested_timestamp(
                timestamp, series.frame.index, series.key
            )
            if point in series.frame.index:
                matches.append(series)
        return tuple(matches)

    def _matching_blueprint_indexes(self, contract: ibi.Contract) -> set[int]:
        """Return indexes of registered blueprints matching ``contract``."""

        return {
            index
            for index, blueprint in enumerate(self._blueprints)
            if _blueprint_matches(blueprint, contract)
        }

    def _build_timestamps(
        self, series: Sequence[StoredSeries]
    ) -> tuple[ReplayTimestamp, ...]:
        """Build the bounded union clock while retaining full source frames."""

        categories = {_index_category(item.frame.index) for item in series}
        if len(categories) > 1:
            raise InvalidBacktestDataError(
                "One replay cannot mix daily date indexes and intraday datetime indexes"
            )
        if categories:
            category = next(iter(categories))
            _validate_bound_category(self._start, category, "start")
            _validate_bound_category(self._end, category, "end")
        timestamps = {point for item in series for point in item.frame.index}
        if self._start is not None:
            timestamps = {point for point in timestamps if point >= self._start}
        if self._end is not None:
            timestamps = {point for point in timestamps if point <= self._end}
        return tuple(sorted(timestamps))


async def load_backtest_data(
    store: BacktestDataStore,
    blueprints: Iterable[ibi.Contract] = (),
    *,
    start: ReplayTimestamp | None = None,
    end: ReplayTimestamp | None = None,
) -> BacktestDataRepository:
    """Create and load a replay repository in one async call.

    Args:
        store: Read-only async datastore for one dataloader library.
        blueprints: Optional contracts used to select stored series.
        start: Optional first replay timestamp.
        end: Optional final replay timestamp.

    Returns:
        A loaded :class:`BacktestDataRepository`.
    """

    repository = BacktestDataRepository(store, start=start, end=end)
    repository.register(*blueprints)
    return await repository.load()


def _contract_from_metadata(key: str, metadata: Mapping[str, Any]) -> ibi.Contract:
    """Reconstruct one ``Contract`` from top-level Arctic metadata."""

    values = {
        name: value
        for name, value in metadata.items()
        if name in _CONTRACT_FIELDS and value is not None
    }
    try:
        contract = ibi.Contract.create(**values)
    except (TypeError, ValueError) as exc:
        raise InvalidBacktestDataError(
            f"Collection {key!r} contains invalid contract metadata"
        ) from exc
    if not contract.secType:
        raise InvalidBacktestDataError(
            f"Collection {key!r} metadata has no contract secType"
        )
    if not contract.localSymbol and not contract.symbol:
        raise InvalidBacktestDataError(
            f"Collection {key!r} metadata has no contract symbol"
        )
    if not contract.conId:
        raise MissingContractMetadataError(
            f"Collection {key!r} metadata has no nonzero contract conId"
        )
    if contract.secType == "FUT":
        expiry = str(contract.lastTradeDateOrContractMonth or "")
        if len(expiry) != 8 or not expiry.isdigit():
            raise MissingContractMetadataError(
                f"Futures collection {key!r} requires an exact YYYYMMDD "
                "lastTradeDateOrContractMonth"
            )
        try:
            datetime.strptime(expiry, "%Y%m%d")
        except ValueError as exc:
            raise InvalidBacktestDataError(
                f"Futures collection {key!r} has invalid expiry {expiry!r}"
            ) from exc
    return contract


def _contract_details_values(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return stored ``ContractDetails`` fields with legacy aliases normalized."""

    values = {
        name: value
        for name, value in metadata.items()
        if name in _CONTRACT_DETAILS_FIELDS and name != "contract"
    }
    for name, value in metadata.items():
        canonical = _DETAIL_ALIASES.get(name, name)
        if (
            canonical in _CONTRACT_DETAILS_FIELDS
            and canonical != "contract"
            and canonical not in values
        ):
            values[canonical] = value
    if "minTick" in values and values["minTick"] is not None:
        try:
            min_tick = float(values["minTick"])
        except (TypeError, ValueError) as exc:
            raise InvalidBacktestDataError(
                "Contract metadata minTick must contain a real number"
            ) from exc
        if not math.isfinite(min_tick) or min_tick < 0:
            raise InvalidBacktestDataError(
                "Contract metadata minTick must be finite and non-negative"
            )
        values["minTick"] = min_tick
    return values


def _series_expiry_key(series: StoredSeries) -> tuple[str, str]:
    """Return the deterministic expiry/key ordering for blueprint details."""

    return series.contract.lastTradeDateOrContractMonth, series.key


def _metadata_candidates(field: str) -> tuple[str, ...]:
    """Return a metadata field and all of its canonical or legacy aliases."""

    canonical = _DETAIL_ALIASES.get(field, field)
    aliases = tuple(
        alias for alias, target in _DETAIL_ALIASES.items() if target == canonical
    )
    return tuple(dict.fromkeys((field, canonical, *aliases)))


def _validate_frame(key: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Copy and validate one replay dataframe without changing stored data."""

    if not isinstance(frame, pd.DataFrame):
        raise InvalidBacktestDataError(
            f"Backtest collection {key!r} did not return a DataFrame"
        )
    missing = _REQUIRED_BAR_COLUMNS - set(frame.columns)
    if missing:
        raise InvalidBacktestDataError(
            f"Backtest collection {key!r} is missing OHLC column(s): "
            + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise InvalidBacktestDataError(f"Backtest collection {key!r} has no bar rows")
    for column in sorted(_REQUIRED_BAR_COLUMNS):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise InvalidBacktestDataError(
                f"Backtest collection {key!r} has non-numeric {column!r} values"
            )
    for column in sorted(_REQUIRED_PRICE_COLUMNS):
        if frame[column].isna().any():
            raise InvalidBacktestDataError(
                f"Backtest collection {key!r} has missing {column!r} values"
            )
        if not frame[column].map(lambda value: math.isfinite(float(value))).all():
            raise InvalidBacktestDataError(
                f"Backtest collection {key!r} has non-finite {column!r} values"
            )
    result = frame.copy()
    if result.index.has_duplicates:
        raise InvalidBacktestDataError(
            f"Backtest collection {key!r} has duplicate timestamps"
        )
    if not result.index.is_monotonic_increasing:
        result = result.sort_index()
    _index_category(result.index, key)
    return result


def _index_category(index: pd.Index, key: str = "series") -> str:
    """Return ``date`` or ``datetime`` after validating an index category."""

    if isinstance(index, pd.DatetimeIndex):
        if index.tz is None:
            raise InvalidBacktestDataError(
                f"Intraday collection {key!r} must have a timezone-aware index"
            )
        try:
            is_utc = all(point.utcoffset() == pd.Timedelta(0) for point in index)
        except (AttributeError, TypeError) as exc:
            raise InvalidBacktestDataError(
                f"Collection {key!r} has an invalid datetime index"
            ) from exc
        if not is_utc:
            raise InvalidBacktestDataError(
                f"Intraday collection {key!r} must use UTC timestamps"
            )
        return "datetime"
    if all(
        isinstance(point, date) and not isinstance(point, datetime) for point in index
    ):
        return "date"
    if len(index) == 0:
        raise InvalidBacktestDataError(f"Backtest collection {key!r} has no timestamps")
    raise InvalidBacktestDataError(
        f"Collection {key!r} must use daily dates or aware UTC datetimes"
    )


def _normalize_bound(
    point: ReplayTimestamp | None, label: str
) -> ReplayTimestamp | None:
    """Validate a replay bound and normalize aware datetimes to UTC."""

    if point is None:
        return None
    if isinstance(point, datetime):
        if point.tzinfo is None or point.utcoffset() is None:
            raise InvalidBacktestDataError(
                f"Intraday replay {label} must be timezone-aware"
            )
        return point.astimezone(timezone.utc)
    if isinstance(point, date):
        return point
    raise TypeError(f"Replay {label} must be a date, datetime, or None")


def _validate_bound_pair(
    start: ReplayTimestamp | None, end: ReplayTimestamp | None
) -> None:
    """Validate the ordering and common category of optional replay bounds."""

    if start is None or end is None:
        return
    if isinstance(start, datetime) != isinstance(end, datetime):
        raise InvalidBacktestDataError(
            "Replay start and end must use the same date category"
        )
    if start > end:
        raise InvalidBacktestDataError("Replay start must not be after replay end")


def _validate_bound_category(
    point: ReplayTimestamp | None, category: str, label: str
) -> None:
    """Ensure a replay bound uses the category of the loaded series."""

    if point is None:
        return
    if category == "datetime" and not isinstance(point, datetime):
        raise InvalidBacktestDataError(
            f"Intraday replay {label} must be an aware datetime"
        )
    if category == "date" and isinstance(point, datetime):
        raise InvalidBacktestDataError(f"Daily replay {label} must be a date")


def _normalize_requested_timestamp(
    point: ReplayTimestamp, index: pd.Index, key: str
) -> ReplayTimestamp:
    """Normalize and category-check a point used against one series index."""

    # Every StoredSeries index is fully validated once during load(). Repeating
    # _index_category() here would scan an entire DatetimeIndex for every bar
    # lookup and turn an otherwise linear replay into quadratic work.
    category = "datetime" if isinstance(index, pd.DatetimeIndex) else "date"
    normalized = _normalize_bound(point, "timestamp")
    assert normalized is not None
    if category == "datetime" and not isinstance(normalized, datetime):
        raise InvalidBacktestDataError(
            f"Intraday collection {key!r} requires a datetime replay point"
        )
    if category == "date" and isinstance(normalized, datetime):
        raise InvalidBacktestDataError(
            f"Daily collection {key!r} requires a date replay point"
        )
    return normalized


def _blueprint_matches(blueprint: ibi.Contract, candidate: ibi.Contract) -> bool:
    """Return whether a persisted contract belongs to a registered blueprint."""

    if blueprint.secType == "CONTFUT":
        if candidate.secType != "FUT":
            return False
        return _matching_fields(blueprint, candidate, _CONT_FUTURE_IDENTITY_FIELDS)
    if blueprint.conId:
        return bool(candidate.conId and blueprint.conId == candidate.conId)
    if blueprint.secType and blueprint.secType != candidate.secType:
        return False
    return _matching_nondefault_fields(blueprint, candidate)


def _metadata_may_match_blueprint(
    blueprint: ibi.Contract, metadata: Mapping[str, Any]
) -> bool:
    """Return whether raw metadata could belong to ``blueprint``.

    This is a conservative prefilter used before strict Contract reconstruction.
    A key is excluded only when a present, type-compatible identity value proves
    that it cannot match. Missing or malformed values remain candidates so their
    metadata errors are not hidden.
    """

    if blueprint.secType == "CONTFUT":
        if _metadata_field_conflicts(metadata, "secType", "FUT"):
            return False
        return not any(
            _metadata_field_conflicts(metadata, name, getattr(blueprint, name))
            for name in _CONT_FUTURE_IDENTITY_FIELDS
            if getattr(blueprint, name) != getattr(_DEFAULT_CONTRACT, name)
        )
    if blueprint.conId:
        return not _metadata_field_conflicts(metadata, "conId", blueprint.conId)
    if blueprint.secType and _metadata_field_conflicts(
        metadata, "secType", blueprint.secType
    ):
        return False
    return not any(
        _metadata_field_conflicts(metadata, name, getattr(blueprint, name))
        for name in _CONTRACT_FIELD_NAMES
        if name not in _NON_IDENTITY_CONTRACT_FIELDS
        and getattr(blueprint, name) != getattr(_DEFAULT_CONTRACT, name)
    )


def _metadata_field_conflicts(
    metadata: Mapping[str, Any], name: str, expected: Any
) -> bool:
    """Return whether one persisted identity value proves a mismatch."""

    if name not in metadata:
        return False
    actual = metadata[name]
    if actual is None or actual == "":
        return False
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and actual == 0
    ):
        return False
    if type(actual) is not type(expected):
        return False
    return actual != expected


def _lookup_matches(requested: ibi.Contract, candidate: ibi.Contract) -> bool:
    """Return whether a loaded concrete contract satisfies a lookup contract."""

    if requested.conId:
        return bool(candidate.conId and requested.conId == candidate.conId)
    if requested.secType == "CONTFUT":
        return _blueprint_matches(requested, candidate)
    return _blueprint_matches(requested, candidate)


def _matching_fields(
    blueprint: ibi.Contract, candidate: ibi.Contract, names: Iterable[str]
) -> bool:
    """Compare every non-default blueprint field named by ``names``."""

    for name in names:
        expected = getattr(blueprint, name, None)
        if expected not in (None, "", 0) and expected != getattr(candidate, name, None):
            return False
    return True


def _matching_nondefault_fields(
    blueprint: ibi.Contract, candidate: ibi.Contract
) -> bool:
    """Compare every specified Contract identity field on a blueprint.

    This includes derivative identity such as expiry, strike, right, and
    multiplier. Broker request controls such as ``includeExpired`` are not
    contract identity and are deliberately ignored.
    """

    for name in _CONTRACT_FIELD_NAMES:
        if name in _NON_IDENTITY_CONTRACT_FIELDS:
            continue
        expected = getattr(blueprint, name)
        if expected == getattr(_DEFAULT_CONTRACT, name):
            continue
        if expected != getattr(candidate, name):
            return False
    return True


def _row_to_bar(timestamp: object, row: pd.Series) -> ibi.BarData:
    """Convert one validated dataframe row into an ``ib_insync.BarData``."""

    values: dict[str, Any] = {"date": timestamp}
    for field in fields(ibi.BarData):
        if field.name == "date" or field.name not in row.index:
            continue
        value = row[field.name]
        if pd.isna(value):
            values[field.name] = math.nan
            continue
        values[field.name] = value.item() if hasattr(value, "item") else value
    return ibi.BarData(**values)


__all__ = [
    "AmbiguousBacktestDataError",
    "BacktestDataError",
    "BacktestDataNotLoadedError",
    "BacktestDataRepository",
    "BacktestDataStore",
    "InvalidBacktestDataError",
    "MissingBacktestDataError",
    "MissingContractMetadataError",
    "ReplayTimestamp",
    "StoredSeries",
    "load_backtest_data",
]
