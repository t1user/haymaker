import pandas as pd  # type: ignore
import numpy as np
from numpy.linalg import lstsq as lst
from numba import jit, prange  # type: ignore
from typing import Optional, Union, Literal, Tuple, Callable


# Swing trading signals ###


@jit(nopython=True)
def _swing(data: np.ndarray, f: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """
    Numba optimized implementation for swing indicators.  Should not
    be used directly, but via swing function which accepts pandas
    objects.

    Args:
    -----

    data: array with two columns, first represents high prices, second
    low prices

    f: filter for swings (price movement required to establish swing
    reversal); must be given as an array with equal number of rows as
    data; can be constant for every row or vary depending on
    percentage of price, ATR, or other (calculation of f for every
    data row must be done outside of this function)

    margin: size of move (in price points) past previous opposite
    swing to determine trade signal; can be constant or calculated
    based on any criteria outside of this function

    Returns:
    --------

    Numpy array as described in docstring to function swing.
    """

    state = np.ones((data.shape[0], 1), dtype=np.int8)
    extreme = np.zeros(state.shape)
    extreme[0] = data[0, 0]
    pivot = np.zeros(state.shape)
    pivot[0] = data[0, 0]
    signal = np.ones(state.shape, dtype=np.int8)
    signal[0] = 1
    max_pivot = np.array([data[0, 0]])
    min_pivot = np.array([data[0, 1]])
    max_list = np.zeros(state.shape)
    min_list = np.zeros(state.shape)

    for i, row in enumerate(data):
        # print(f'{i} row: {row}, extreme: {extreme[i-1]}')
        if i == 0:
            continue

        if state[i - 1, 0] == 1:
            if row[0] > (max_pivot[0] + margin[i]):
                signal[i] = 1
            else:
                signal[i] = signal[i - 1]

            if (extreme[i - 1, 0] - row[1]) > f[i]:
                state[i] = -1
                extreme[i] = row[1]
                pivot[i] = extreme[i - 1]
                max_pivot = pivot[i]
                # max_list[i] = max_pivot

            else:
                state[i] = 1
                extreme[i] = max(extreme[i - 1, 0], row[0])
                pivot[i] = pivot[i - 1]

        elif state[i - 1, 0] == -1:
            if row[1] < (min_pivot[0] - margin[i]):
                signal[i] = -1
            else:
                signal[i] = signal[i - 1]

            if np.abs(extreme[i - 1, 0] - row[0]) > f[i]:
                state[i] = 1
                extreme[i] = row[0]
                pivot[i] = extreme[i - 1]
                min_pivot = pivot[i]
                # min_list[i] = min_pivot

            else:
                state[i, 0] = -1
                extreme[i] = min(extreme[i - 1, 0], row[1])
                pivot[i] = pivot[i - 1]

        else:
            raise ValueError("Wrong state value!")
        min_list[i] = min_pivot
        max_list[i] = max_pivot
    return np.concatenate((state, extreme, pivot, signal, max_list, min_list), axis=1)


def swing(
    data: pd.DataFrame,
    f: Union[float, np.ndarray, pd.Series],
    margin: Optional[Union[float, pd.Series]] = None,
    output_as_tuple: bool = True,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Simulate swing trading signals and return generated output series.

    User interface allowing to use numba optimized _swing function
    passing pandas objects (See _swing docstring).  This should be
    used as entry point to _swing.

    The function needs runway.  It starts at long swing state, which
    maybe incorrect.  To be sure correct state is used, it's best to
    disregard values until first swing change.

    For each new series starts at max of the first bar in series and only
    adjusts to correct value after first state change

    Args:
    -----

    data: must have columns named 'high' and 'low' with high and low
    prices for a given price bar

    f: filter to determine if swing changed direction, if given as
    float will be the same for all data points, can also be given as
    pd.Series or one collumn np.array with values for every data point
    (which can be pre-calculated based on ATR, vol, % of price or any
    other criteria)

    margin: how much previous swing pivot has to be breached to
    establish swing reversal, expressed in multiples of f

    output_as_tuple: whether data should be returned as a named tuple
    with each property as a pd.Series or raw numpy array

    Returns:
    --------

    Numpy array with shape [data.shape[0], 6] (if output_as_df is
    False) or original data DataFrame with additional columns.

    Generated columns have following meaning for every row
    corresponding to price data ([np.array column index]/DataFrame
    column name):

    [0]/state - what swing are we in?  (1) for upswing (-1) for
    downswing

    [1]/extreme - current extreme (since last pivot) against which
    filter is applied to determine if swing changed direction;

    [2]/pivot - last swing pivot point, last price at which state
    changed;

    [3]/signal - trade signal, generated when opposite swing pivot is
    breached by margin;

    [4]/last_max - latest upswing pivot

    [5]/last_min - latest downswing pivot
    """
    if isinstance(f, (float, int)):
        f = np.ones((data.shape[0], 1)) * f
    elif isinstance(f, (pd.DataFrame, pd.Series)):
        f = f.to_numpy()
    if margin is None:
        _margin = 0 * f
    elif isinstance(margin, (pd.DataFrame, pd.Series)):
        _margin = margin.to_numpy() * f
    elif isinstance(margin, float):
        _margin = margin * f
    else:
        raise TypeError("Margin must be pd.Series, pd.DataFrame or float")

    _data = data[["high", "low"]].to_numpy()

    output = _swing(_data, f, _margin)

    if output_as_tuple:
        return data.join(
            pd.DataFrame(
                output,
                columns=["state", "extreme", "pivot", "signal", "last_max", "last_min"],
                index=data.index,
            )
        )
    else:
        return output


# Blip to position converter ###


@jit(nopython=True)
def _blip_to_signal_converter(data: np.ndarray, always_on=True) -> np.ndarray:
    """
    Given a column with blips, return a column with signals
    resulting from those blips.

    Args:
    -----

    data: one column with values (-1, 0, 1)
    always_on: if True, blip oposite to current signal, generates oposite signal;
               if False, blip oposite to current signal, cancels existing signal
               (returns 0)

    Returns:
    --------

    Single column array representing signal from blips.
    """
    # Don't replace minimum and maximum with clip - doesn't work with numba!!!

    state = np.ones(data.shape, dtype=np.int8)

    for i, row in enumerate(data):
        if i == 0:
            state[i] = row
        else:
            if always_on:
                state[i] = row or state[i - 1]
            else:
                state[i] = np.maximum(np.minimum((state[i - 1] + row), 1), -1)

    return state


# In-Out signal unifier ###


@jit(nopython=True)
def _in_out_signal_unifier(data: np.ndarray) -> np.ndarray:
    """Given an array with two columns, one with entry and one with close
    signal, return an array with signal resulting from combining those
    two signals.

    THIS IS FOR BLIPS???? WORKS FOR BOTH: SIGNALS AND BLIPS? -
    !!!!!TEST!!!!!!  if entry is a signal, it will immediately reenter
    position after close-out, right?

    !!!!!!!WORKS ONLY FOR BLIPS!!!!!

    Args:
    -----

    data: must have two columns: [0] - entry signal/blip [1] - close signal/blip

    Returns:
    --------

    Numpy array of shape [data.shape[0], 1] with signal resulting
    from applying entry and close signals/blips (1: long, 0: not on the market,
    -1: short).

    """

    # Don't replace min and max with clip - doesn't work with numba!!!

    state = np.zeros((data.shape[0], 1), dtype=np.int8)

    for i, row in enumerate(data):
        if i == 0:
            state[i] = row[0]
        else:
            if state[i - 1] != 0:
                state[i] = (
                    np.maximum(np.minimum((state[i - 1] + row[1]), 1), -1)
                ).astype(np.int8)
                # previous version added also row[0] here
            else:
                state[i] = row[0]

    return state


# volume grouper ###


@jit(nopython=True)
def _volume_grouper(volume: np.ndarray, target: int) -> np.ndarray:
    aggregator = 0
    label = 0
    labels = np.zeros(volume.shape)
    for i, row in enumerate(volume):
        aggregator += row
        labels[i] = label
        if aggregator >= target:
            aggregator = 0
            label += 1
    return labels


def volume_grouper(
    df: pd.DataFrame, target, label: Literal["left", "right"] = "right"
) -> pd.DataFrame:

    if not set(["open", "high", "low", "close", "volume", "barCount"]).issubset(
        set(df.columns)
    ):
        raise ValueError(
            "df must have all of the columns: 'open', 'high', 'low',"
            " 'close', 'volume', 'barCount'"
        )
    if label == "left":
        _label = "first"
    elif label == "right":
        _label = "last"
    else:
        raise ValueError("label must be either 'left' or 'right'")

    df = df.copy().reset_index()
    df["labels"] = _volume_grouper(df.volume.to_numpy(), target)
    return (
        df.groupby("labels")
        .agg(
            {
                "date": _label,
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "barCount": "sum",
            }
        )
        .set_index("date")
    )


# pivot ###


@jit(nopython=True)
def _pivot(
    price: np.ndarray, f: np.ndarray, initial_state: Literal[-1, 1] = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.ones((price.shape[0]), dtype=np.int8)
    state[0] = initial_state
    extreme_list = []
    extreme = price[0]
    extreme_index = 0

    for i, p in enumerate(price):

        # check for change of state
        if (state[i - 1] * p) < (state[i - 1] * (extreme - (state[i - 1] * f[i - 1]))):
            state[i] = -state[i - 1]
            extreme_list.append(state[i - 1] * extreme_index)
            extreme_index = i
            extreme = p
        else:
            state[i] = state[i - 1]

            # check for new extreme
            extreme_diff = state[i - 1] * np.maximum(-state[i - 1] * (extreme - p), 0)
            if extreme_diff != 0:
                extreme_index = i
                extreme = extreme + extreme_diff

    extremes = np.array(extreme_list)
    mins = -extremes[extremes < 0]
    maxes = extremes[extremes > 0]
    return state, mins, maxes


def pivot(
    data: Union[pd.DataFrame, pd.Series],
    f: Union[pd.Series, float, int],
    initial_state: Literal[-1, 1] = 1,
    return_type: int = 3,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
    """Calculate local peaks and troughs in a time series.

    Given a filter f and a price series, assume that a price move away
    from local extreme by at least f points constitues a change in
    trend and record this local extreme.

    Even if df given, only close price will be taken into account
    (highs and lows ignored).

    WARNING: inappropriate use will lead to forward data snooping,
    eg. in 'down' state, the exact minimum poin will be unknown until
    state changes. So in 'down' state only last max value/index should
    be used and in 'up' state only last min. Both could be use for
    graphical representations but not for making trading decisions!
    For both forward-safe extremes use <swing>.

    Args:
    ---------

    data: price series; must be a Series or DataFrame with column 'close'

    f: trend filter; number of points that the price has to move away from last
    extreme for the trend to be considered changed

    initial_state: at the beginning of price series direction of trend
    is unknown so a side has to be picked arbitrarily; it doesn't
    matter much because trend will be corrected into actual state
    after runway period

    return_type: how data is to be returned, values:

    1 - tuple of 3 ndarrays with mins, maxes, state; mins, maxes
    contain indexes of respective extremes; state is a series that
    matches original data index, 1 means uptrand, -1 downtrend

    2 - DataFrame with 5 columns: mins, maxes, state, mins_, maxes_;

    mins, maxes: contain True at the index point of respective
    extreme, False otherwise;

    state: contains -1 or 1 depending on the trend at given point;

    3 - original DataFrame (or DataFrame with original price series)
    with additial 3 columns with names and meanings same as in option
    2 and additionally 4 columns with meaning:

    min_val, maxe_val: for every point the value of last min or max

    min_index, max_index: indexes of last min or max


    Returns:
    ---------

    Return format determined by return_type argument.

    """

    if isinstance(data, pd.Series):
        data = pd.DataFrame({"close": data})
    elif isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise ValueError("DataFrame must have column 'close'")
    else:
        raise TypeError("data must be a Series or DataFrame")

    price = data["close"].to_numpy()

    if isinstance(f, pd.Series):
        _f = f.to_numpy()
    elif isinstance(f, (float, int)):
        _f = np.ones((data.shape[0])) * f
    else:
        raise TypeError(f"Wrong type: {f}")

    state, mins, maxes = _pivot(price, _f, initial_state)

    if return_type == 1:
        return state, mins, maxes

    df = data.copy()

    df[["mins", "maxes"]] = False
    df.iloc[maxes, df.columns.get_loc("maxes")] = True
    df.iloc[mins, df.columns.get_loc("mins")] = True

    df["date"] = df.index
    df[["min_index", "max_index"]] = np.nan
    df["min_index"] = df["date"].mask(~df["mins"]).ffill()
    df["max_index"] = df["date"].mask(~df["maxes"]).ffill()
    del df["date"]

    df[["mins", "maxes"]] = df[["mins", "maxes"]].replace(False, np.nan)
    df["min_val"] = (df["mins"] * df["close"]).ffill()
    df["max_val"] = (df["maxes"] * df["close"]).ffill()

    df["state"] = state

    if return_type == 2:
        return df[["mins", "maxes", "state"]]
    else:
        return df


def clear_runway(df: pd.DataFrame, initial_state: Literal[-1, 1] = 1) -> pd.DataFrame:
    """For swing and pivot function - clear initial initialization period data."""
    data = df[df["state"] == -initial_state]
    if len(data) > 0:
        start = data.index[0]
        assert isinstance(start, int)
        data = df[start:][1:]
    return data


# pivot indicator ###


# --------------
# These indicator functions can be passed as <func> parameter to <pivot_indicator>
# --------------
#
# signature of indicator function:
#
# @jit(nopython=True)
# def func(x: np.ndarray, y: np.adarray) -> Tuple[float, float, float]:
#     return a, b, c

# where x is an index vector, y i vector wtih prices
# and a, b, c are floats
# if less than 3 values are to be returned, 'empty' values should
# be set to 0.0
#
# Functions must be numba decorated.


@jit(nopython=True)
def regression(
    x: np.ndarray, y: np.ndarray, fit_intercept=False
) -> Tuple[float, float, float]:
    """Return regression slope, intercept and value, where value is prediction by
    regression fitted function.

    if fit_intercept is False, intercept will be a zero vector.
    """

    x_ = x - x[0]
    y_ = np.log(y / y[0])

    if fit_intercept:
        x_ = np.vstack((x_, np.ones(x_.shape[0]))).T
    else:
        x_ = np.vstack((x_, np.zeros(x_.shape[0]))).T

    slope, intercept = lst(x_, y_)[0]

    value = np.exp((slope * x_[-1, 0]) + intercept) * y[0]

    return slope, intercept, value


@jit(nopython=True)
def regression_with_intercept(x: np.ndarray, y: np.ndarray):
    """By default regression will fit function without intercept. This is
    a shortcut that allows to fit a function with intercept (numba doesn't
    accept partial objects).
    """
    return regression(x, y, True)


regression_no_intercept = regression


@jit(nopython=True)
def mean_sd_z(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    mean = np.mean(y)
    std = np.std(y)
    return mean, std, (y[-1] - mean) / std


# --------------
# End of indicator functions
# --------------


@jit(nopython=True, parallel=True)
def _process_chunk(arr, offset, func):

    out = np.zeros(
        (arr[offset:].shape[0], 3),
    )

    for i in prange(offset, arr.shape[0]):
        x, y = arr[: i + 1].T
        out[i - offset] = func(x, y)
    return out


@jit(nopython=True)
def _pivot_indicator(indexes, arr, func):
    out = []
    for pivot, start, stop in indexes:
        offset = start - pivot
        out.append(_process_chunk(arr[pivot:stop], offset, func))
    return out


def pivot_indicator(df: pd.DataFrame, func: Callable) -> pd.DataFrame:
    """
    Calculates expanding <func> since the last pivot.

    This is an interface function for farious numba enhanced no-python functions.

    df is the result of pivot function with desired parameters. It must have columns:
    state, maxes, mins.

    Usage example:

    d = df.copy()
    d['atr'] = downsampled_atr(d, 60*23)
    # always pass the result of pivot function with desired parameters
    d = pivot(d, d['atr'])

    dd = pivot_indicator(d, regression)
    dd.columns = ['slope', 'intercept', 'value']

    or:

    dd = pivot_indicator(d, mean_sd_z)
    dd.columns = ['mean', 'std', 'z']

    """
    d = df.copy()
    # ind is row index
    d["ind"] = np.arange(d.shape[0])
    # index of last max
    d["max_ind"] = (d["ind"] * d["maxes"]).ffill()
    # index of last min
    d["min_ind"] = (d["ind"] * d["mins"]).ffill()
    # index of last pivot
    d["pivot_ind"] = ((d["state"] == 1) * d["min_ind"]) + (
        (d["state"] == -1) * d["max_ind"]
    )
    # did the state change True or False
    d["state_change"] = d["state"] != d["state"].shift()

    state_and_pivot = d[d["state_change"] == True].loc[:, ["ind", "pivot_ind"]]  # noqa
    state_and_pivot["start"] = state_and_pivot["ind"]
    state_and_pivot["stop"] = state_and_pivot["start"].shift(-1).fillna(len(d))

    # indexes of pivot, state start index and state stop index
    state_and_pivot = state_and_pivot.iloc[:, 1:].dropna().astype(int)

    df_start = state_and_pivot["start"].iloc[0]

    _state_and_pivot = state_and_pivot.to_numpy()

    # index and close prices
    arr = d[["ind", "close"]].values
    out = _pivot_indicator(_state_and_pivot, arr, func)
    indicator = np.concatenate(out)

    index = d.iloc[df_start:].index
    indicator = pd.DataFrame(indicator, index=index)
    return indicator


# box chart ###
