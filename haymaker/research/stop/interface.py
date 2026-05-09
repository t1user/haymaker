import datetime as dt
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from ..signal_converters import pos_trans, pos_trans_array
from .python_impl import StopMode, StopParams, param_factory

DistanceLike = float | int | pd.Series
ScheduledCloseLike = dt.time | tuple[int, ...] | pd.Series | None
TimedeltaLike = dt.timedelta | pd.Timedelta
TransactionMethod = Literal["numpy", "pandas"]


class PreparedData(NamedTuple):
    first: np.ndarray
    second: np.ndarray
    high: np.ndarray
    low: np.ndarray
    price: np.ndarray
    distance: float | np.ndarray
    scheduled_close: np.ndarray
    row_index: pd.Index
    use_blip: bool


def _validate_inputs(
    df: pd.DataFrame,
    distance: DistanceLike,
    price_column: str,
    scheduled_close: ScheduledCloseLike,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("df must have columns: 'high', 'low'")
    if "position" not in df.columns and "blip" not in df.columns:
        raise ValueError("df must have either column 'position' or 'blip'")
    if price_column not in df.columns:
        raise ValueError(f"'{price_column}' indicated as price column, but not in df")
    if not isinstance(distance, (pd.Series, float, int)):
        raise ValueError(f"distance must be series or number, not {type(distance)}")
    if isinstance(distance, pd.Series) and not distance.index.equals(df.index):
        raise ValueError(
            "distance Series must have the same index as df. Align or upsample "
            "distance explicitly before calling stop_loss."
        )
    if isinstance(scheduled_close, pd.Series):
        if not scheduled_close.index.equals(df.index):
            raise ValueError("scheduled_close Series must have the same index as df.")
        if scheduled_close.isna().any():
            raise ValueError("scheduled_close Series must not contain n/a values.")
    elif scheduled_close is not None and not isinstance(
        scheduled_close, (dt.time, tuple)
    ):
        raise ValueError(
            "scheduled_close must be datetime.time, tuple, pandas Series, or None."
        )


def _distance_data(
    distance: DistanceLike,
) -> float | np.ndarray:
    if isinstance(distance, pd.Series):
        return distance.to_numpy(dtype=np.float64, copy=False)
    return float(distance)


def _scheduled_close_time(value: dt.time | tuple[int, ...]) -> dt.time:
    if isinstance(value, dt.time):
        return value
    try:
        assert isinstance(value, tuple)
        return dt.time(*value)  # type: ignore
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "scheduled_close tuple must be acceptable by datetime.time."
        ) from exc
    raise ValueError("scheduled_close tuple must be acceptable by datetime.time.")


def _scheduled_close_data(
    scheduled_close: ScheduledCloseLike, index: pd.Index
) -> np.ndarray:
    if scheduled_close is None:
        return np.zeros(len(index), dtype=np.bool_)
    if isinstance(scheduled_close, pd.Series):
        return scheduled_close.to_numpy(dtype=np.bool_, copy=False)
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("scheduled_close as time or tuple requires a DatetimeIndex.")
    close_time = _scheduled_close_time(scheduled_close)
    if close_time.tzinfo is None:
        return index.time == close_time
    if index.tz is None:
        raise ValueError(
            "timezone-aware scheduled_close requires a timezone-aware DatetimeIndex."
        )
    index_in_close_timezone = index.tz_convert(close_time.tzinfo)
    return index_in_close_timezone.time == close_time.replace(tzinfo=None)


def _as_positive_timedelta(value: TimedeltaLike, name: str) -> pd.Timedelta:
    try:
        delta = pd.Timedelta(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid timedelta.") from exc
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        raise ValueError(f"{name} must be a positive timedelta.")
    return delta


def _data_index(data: pd.DataFrame | pd.Series | pd.Index) -> pd.Index:
    if isinstance(data, pd.Index):
        return data
    return data.index


def _infer_bar_duration(
    index: pd.DatetimeIndex, session_gap: pd.Timedelta | None
) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("bar_duration must be provided for fewer than two rows.")
    deltas = index[1:] - index[:-1]
    positive_deltas = deltas[deltas > pd.Timedelta(0)]
    if session_gap is not None:
        positive_deltas = positive_deltas[positive_deltas <= session_gap]
    if len(positive_deltas) == 0:
        raise ValueError("Could not infer bar_duration from index.")
    return positive_deltas.to_series().mode().iloc[0]


def before_close(
    data: pd.DataFrame | pd.Series | pd.Index,
    offset: TimedeltaLike,
    *,
    session_gap: TimedeltaLike | None = None,
    bar_duration: TimedeltaLike | None = None,
) -> pd.Series:
    """
    Build a scheduled-close mask for bars before each inferred session end.

    The helper is intended for left-labeled bars, where each timestamp
    marks the start of the bar. It infers each session's end as the last
    bar timestamp plus ``bar_duration``. Rows whose timestamps are at or
    after ``session_end - offset`` are marked ``True`` through the final
    bar of that session.

    Args:
    -----
    data:
        DataFrame, Series, or Index whose index is used to infer sessions.
        The index must be a strictly increasing ``pd.DatetimeIndex``.

    offset:
        Positive timedelta before inferred session end where the close
        window begins.

    session_gap:
        Positive timedelta. A gap larger than this value starts a new
        session. If omitted, the threshold is inferred as the larger of
        one hour and three times the inferred ``bar_duration``.

    bar_duration:
        Positive timedelta representing one bar's duration. If omitted,
        it is inferred as the most common positive index step, ignoring
        gaps larger than ``session_gap`` when ``session_gap`` is provided.

    Returns:
    --------
    pd.Series
        Boolean mask named ``"scheduled_close"``. It can be passed
        directly to ``stop_loss(..., scheduled_close=...)``.
    """

    index = _data_index(data)
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("before_close requires a DatetimeIndex.")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("before_close requires a strictly increasing index.")
    if len(index) == 0:
        return pd.Series(
            np.zeros(len(index), dtype=np.bool_),
            index=index,
            name="scheduled_close",
        )

    close_offset = _as_positive_timedelta(offset, "offset")
    gap = (
        _as_positive_timedelta(session_gap, "session_gap")
        if session_gap is not None
        else None
    )
    duration = (
        _as_positive_timedelta(bar_duration, "bar_duration")
        if bar_duration is not None
        else _infer_bar_duration(index, gap)
    )
    if gap is None:
        gap = max(pd.Timedelta(hours=1), duration * 3)

    deltas = index[1:] - index[:-1]
    session_end_locs = np.concatenate((np.flatnonzero(deltas > gap), [len(index) - 1]))
    row_locs = np.arange(len(index))
    row_session_ids = np.searchsorted(session_end_locs, row_locs, side="left")

    index_ns = index.to_numpy(dtype="datetime64[ns]", copy=False).astype(
        np.int64, copy=False
    )
    session_end_ns = index_ns[session_end_locs[row_session_ids]] + duration.value
    close_window_start_ns = session_end_ns - close_offset.value
    close_mask = index_ns >= close_window_start_ns

    return pd.Series(close_mask, index=index, name="scheduled_close")


def _position_data(
    df: pd.DataFrame,
    price_column: str,
    distance: DistanceLike,
    scheduled_close: np.ndarray,
    transaction_method: TransactionMethod,
) -> PreparedData:
    position_series = df["position"]
    position = position_series.to_numpy(dtype=np.int8, copy=False)

    if transaction_method == "numpy":
        transaction = pos_trans_array(position)
    else:
        transaction = pos_trans(position_series).to_numpy(dtype=np.int8, copy=False)

    return PreparedData(
        first=position,
        second=transaction,
        high=df["high"].to_numpy(dtype=np.float64, copy=False),
        low=df["low"].to_numpy(dtype=np.float64, copy=False),
        price=df[price_column].to_numpy(dtype=np.float64, copy=False),
        distance=_distance_data(distance),
        scheduled_close=scheduled_close,
        row_index=df.index,
        use_blip=False,
    )


def _blip_data(
    df: pd.DataFrame,
    price_column: str,
    distance: DistanceLike,
    scheduled_close: np.ndarray,
) -> PreparedData:
    blip = df["blip"].shift().fillna(0).astype(int)
    close_blip = (
        df["close_blip"].shift().fillna(0).astype(int)
        if "close_blip" in df.columns
        else blip
    )
    return PreparedData(
        first=blip.to_numpy(dtype=np.int8, copy=False),
        second=close_blip.to_numpy(dtype=np.int8, copy=False),
        high=df["high"].to_numpy(dtype=np.float64, copy=False),
        low=df["low"].to_numpy(dtype=np.float64, copy=False),
        price=df[price_column].to_numpy(dtype=np.float64, copy=False),
        distance=_distance_data(distance),
        scheduled_close=scheduled_close,
        row_index=df.index,
        use_blip=True,
    )


def _prepare_data(
    df: pd.DataFrame,
    distance: DistanceLike,
    price_column: str,
    scheduled_close: ScheduledCloseLike = None,
    transaction_method: TransactionMethod = "numpy",
) -> PreparedData:
    scheduled_close_data = _scheduled_close_data(scheduled_close, df.index)
    if "position" in df.columns:
        return _position_data(
            df, price_column, distance, scheduled_close_data, transaction_method
        )
    return _blip_data(df, price_column, distance, scheduled_close_data)


def _build_output(
    result: np.ndarray, index: pd.Index, price_series: pd.Series
) -> pd.DataFrame:
    out_df = pd.DataFrame(
        result,
        columns=["position", "open_price", "close_price", "stop_price"],
        index=index,
    )
    out_df["position"] = out_df["position"].astype(int, copy=False)
    out_df["bar_price"] = price_series
    return out_df


def stop_loss(
    df: pd.DataFrame,
    distance: float | pd.Series,
    mode: StopMode = "trail",
    tp_multiple: float = 0,
    adjust: tuple[StopMode, float, float] | None = None,
    time_stop: int = 0,
    scheduled_close: ScheduledCloseLike = None,
    price_column: str = "open",
    use_numba: bool = True,
) -> pd.DataFrame:
    """
    Apply stop loss and optionally take profit to a strategy.

    Convert a series with transactions or positions into a series with
    positions resulting from applying a specified type of stop loss.

    Stop loss can be trailing or fixed and it might be automatically
    adjustable to a different stop after a certain gain has been
    achieved.

    Results of the pre-stop/pre-take-profit strategy can be given as
    positions or blips.

    This is the public interface for stop-loss application. It keeps a
    pandas-friendly API while allowing the core execution loop to run
    either through the reference Python implementation or the Numba
    implementation.

    Args:
    -----
    df:
        Input dataframe. It must contain:

        - ``high`` and ``low`` price columns for each processed bar
        - either ``position`` or ``blip`` as the strategy input

        If both ``position`` and ``blip`` are present, ``position``
        takes precedence.

        If ``blip`` is given, ``close_blip`` may also be provided.
        Then ``blip`` is used to open positions and ``close_blip`` to
        close them. If only ``blip`` is given, it is used for both open
        and close decisions. In this case, if there is an existing position,
        ``blip`` in the opposite direction closes existing position without
        openning an opposing one.

        ``position`` is the desired position before the stop loss is applied.
        This function will return the position after the stop loss is applied.

    distance:
        Desired stop-loss distance. This may be:

        - a scalar, if distance is the same at all time points
        - a ``pd.Series``, to provide a different value for every bar

        ``distance`` must already be expressed in the same price units
        as the prices in ``df``. This function does not rescale it. If
        provided as a ``pd.Series``, it must already have the same index
        as ``df``; this function does not align, upsample, or forward-fill
        distance values.

    mode:
        Stop-loss type to apply. Allowed values are ``"fixed"`` and
        ``"trail"``.

    tp_multiple:
        Optional take-profit distance expressed as a multiple of
        ``distance``. For example, if ``tp_multiple=3``, then the
        take-profit trigger is placed at ``3 * distance`` from entry.
        If zero, no take-profit is used.

    adjust:
        Optional stop-adjustment specification as a tuple:

        - ``adjust[0]``: stop-loss type to adjust to, ``"fixed"`` or
          ``"trail"``
        - ``adjust[1]``: trigger distance from entry, expressed as a
          multiple of the original stop distance
        - ``adjust[2]``: adjusted stop distance, expressed as a
          multiple of the original stop distance

        This allows an initial stop to be replaced by another stop
        definition after the trade has moved sufficiently in favor.

    time_stop:
        Optional time-based exit. If greater than zero, the position is
        closed at ``price_column`` after the specified number of
        processed bars.

    scheduled_close:
        Optional scheduled flattening instruction. If given as
        ``datetime.time`` or a tuple accepted by ``datetime.time``, any
        existing position is closed at ``price_column`` on matching
        index times. Naive times match the index's local wall-clock
        time. Timezone-aware times require a timezone-aware index and
        match after converting the index to the supplied time's
        timezone. If given as a ``pd.Series``, it must be a same-index
        boolean mask. Scheduled closes are execution-time events: they
        are not shifted. They close existing positions, suppress new
        opens on the same bar, and are reported as ``close_price``.

    price_column:
        Name of the column containing the price at which non-stopped
        and time-stopped transactions are executed. In the most common
        use case this is ``"open"``, meaning that a bar's decision is
        acted upon at the next bar's open once the relevant signal is
        already known.

    use_numba:
        If ``True``, run the stop engine through the Numba
        implementation. If ``False``, use the reference Python
        implementation. Both paths are intended to produce identical
        results.

    Important timing note:
        ``blip`` is a generated event. It must be recorded on the bar
        where the information first becomes known, not pre-shifted to
        the execution bar. This function converts generated blips into
        the first possible execution bar internally.

        ``position`` is an already executable/held state. It is
        pre-processed by the calling code and must appear on the bar
        where the position already exists. This function does not shift
        positions.

        The calling code must ensure, taking into account the above
        interpretations, that all inputs reflect when the strategy
        could have known the information and when it could first have
        acted on it. Future leakage invalidates research results.

    Returns:
    --------
    pd.DataFrame
        Dataframe with columns:

        - ``position``: resulting position after transactions on the bar
        - ``open_price``: signed price of an opening transaction
        - ``close_price``: signed price of a normal close transaction
        - ``stop_price``: signed price of a stop/take-profit/time-stop
          transaction

        Prices are signed by trade direction to remain consistent with
        downstream research code.
    """

    _validate_inputs(df, distance, price_column, scheduled_close)
    params: StopParams = param_factory(mode, tp_multiple, time_stop, adjust)
    data = _prepare_data(df, distance, price_column, scheduled_close)

    if use_numba:
        from .numba_impl import run_stop_loss as run_stop_loss_numba

        result = run_stop_loss_numba(
            data.first,
            data.second,
            data.high,
            data.low,
            data.distance,
            data.price,
            data.scheduled_close,
            data.use_blip,
            params,
        )
    else:
        from .python_impl import run_stop_loss as run_stop_loss_python

        result = run_stop_loss_python(
            data.first,
            data.second,
            data.high,
            data.low,
            data.distance,
            data.price,
            data.scheduled_close,
            data.use_blip,
            params,
        )

    price_series = df[price_column]
    return _build_output(result, data.row_index, price_series)
