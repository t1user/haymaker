from typing import Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..signal_converters import pos_trans, pos_trans_array
from .python_impl import StopMode, StopParams, param_factory

DistanceLike = Union[float, int, pd.Series]
TransactionMethod = Literal["numpy", "pandas"]


class PreparedData(NamedTuple):
    first: np.ndarray
    second: np.ndarray
    high: np.ndarray
    low: np.ndarray
    price: np.ndarray
    distance: float | np.ndarray
    row_index: pd.Index
    use_blip: bool


def _validate_inputs(
    df: pd.DataFrame,
    distance: DistanceLike,
    price_column: str,
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


def _distance_data(
    distance: DistanceLike,
    index: pd.Index,
) -> float | np.ndarray:
    if isinstance(distance, pd.Series):
        return distance.reindex(index).to_numpy(dtype=np.float64, copy=False)
    return float(distance)


def _position_data(
    df: pd.DataFrame,
    price_column: str,
    distance: DistanceLike,
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
        distance=_distance_data(distance, df.index),
        row_index=df.index,
        use_blip=False,
    )


def _blip_data(
    df: pd.DataFrame,
    price_column: str,
    distance: DistanceLike,
) -> PreparedData:
    blip = df['blip'].shift().fillna(0).astype(int)
    close_blip = df["close_blip"].shift().fillna(0).astype(int
        ) if "close_blip" in df.columns else blip
    return PreparedData(
        first=blip.to_numpy(dtype=np.int8, copy=False),
        second=close_blip.to_numpy(dtype=np.int8, copy=False),
        high=df["high"].to_numpy(dtype=np.float64, copy=False),
        low=df["low"].to_numpy(dtype=np.float64, copy=False),
        price=df[price_column].to_numpy(dtype=np.float64, copy=False),
        distance=_distance_data(distance, df.index),
        row_index=df.index,
        use_blip=True,
    )


def _prepare_data(
    df: pd.DataFrame,
    distance: DistanceLike,
    price_column: str,
    transaction_method: TransactionMethod = "numpy",
) -> PreparedData:
    if "position" in df.columns:
        return _position_data(df, price_column, distance, transaction_method)
    return _blip_data(df, price_column, distance)


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
    distance: Union[float, pd.Series],
    mode: StopMode = "trail",
    tp_multiple: float = 0,
    adjust: Optional[Tuple[StopMode, float, float]] = None,
    time_stop: int = 0,
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
        as the prices in ``df``. This function does not rescale it.

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
        Blip is a signal to be acted upon. It must be given at the bar 
        where is generated (where information first appeared). The system will 
        make sure it is executed at the first possible bar where it's possible
        to act on the information.
        
        Position indicates a state. It's already pre-processed by the calling code.  
        It must appear at the end of the bar where position already exists. 
        This function does not shift positions.  
        
        The calling code must ensure, taking into account the above interpretations,
        that the timing of signals reflects when the strategy could have known about
        them and when it could first have acted on them. It's a fundamental part of
        a research process, introduction of any 'future leakage' invalidates the results.

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

    _validate_inputs(df, distance, price_column)
    params: StopParams = param_factory(mode, tp_multiple, time_stop, adjust)
    data = _prepare_data(df, distance, price_column)

    if use_numba:
        from .numba_impl import run_stop_loss as run_stop_loss_numba

        result = run_stop_loss_numba(
            data.first,
            data.second,
            data.high,
            data.low,
            data.distance,
            data.price,
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
            data.use_blip,
            params,
        )

    price_series = df[price_column]
    return _build_output(result, data.row_index, price_series)
