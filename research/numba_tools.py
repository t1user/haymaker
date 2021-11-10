import pandas as pd  # type: ignore
import numpy as np
from numba import jit  # type: ignore
from typing import Optional, Union


### Swing trading signals ###

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

        if state[i-1, 0] == 1:
            if (row[0] > (max_pivot[0] + margin[i])):
                signal[i] = 1
            else:
                signal[i] = signal[i-1]

            if (extreme[i-1, 0] - row[1]) > f[i]:
                state[i] = -1
                extreme[i] = row[1]
                pivot[i] = extreme[i-1]
                max_pivot = pivot[i]
                # max_list[i] = max_pivot

            else:
                state[i] = 1
                extreme[i] = max(extreme[i-1, 0], row[0])
                pivot[i] = pivot[i-1]

        elif state[i-1, 0] == -1:
            if (row[1] < (min_pivot[0] - margin[i])):
                signal[i] = -1
            else:
                signal[i] = signal[i-1]

            if np.abs(extreme[i-1, 0] - row[0]) > f[i]:
                state[i] = 1
                extreme[i] = row[0]
                pivot[i] = extreme[i-1]
                min_pivot = pivot[i]
                # min_list[i] = min_pivot

            else:
                state[i, 0] = -1
                extreme[i] = min(extreme[i-1, 0], row[1])
                pivot[i] = pivot[i-1]

        else:
            raise ValueError('Wrong state value!')
        min_list[i] = min_pivot
        max_list[i] = max_pivot
    return np.concatenate((state, extreme, pivot, signal, max_list, min_list),
                          axis=1)


def swing(data: pd.DataFrame, f: Union[float, np.ndarray, pd.Series],
          margin: Optional[Union[float, pd.Series]] = None,
          output_as_tuple: bool = True
          ) -> Union[np.ndarray, pd.DataFrame]:
    """
    Simulate swing trading signals and return generated output series.

    User interface allowing to use numba optimized _swing function
    passing pandas objects (See _swing docstring).  This should be
    used as entry point to _swing.

    The function needs runway.  It starts at long swing state, which
    maybe incorrect.  To be sure correct state is used, it's best to
    disregard values until first swing change.

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
    filter is applied to determine if swing changed direction; for
    each new series starts at max of the first bar in series and only
    adjusts to correct value after first state change

    [2]/pivot - last swing pivot point, last price at which state
    changed; for each new series starts at max of the first bar in
    series and only adjusts to correct value after first state change

    [3]/signal - trade signal, generated when opposite swing pivot is
    breached by margin; at every series swing starts at 1 at only
    later adjusts to correct value

    [4]/last_max - latest upswing pivot

    [5]/last_min - latest downswing pivot
    """
    if isinstance(f, (float, int)):
        f = np.ones((data.shape[0], 1)) * f
    elif isinstance(f, (pd.DataFrame, pd.Series)):
        f = f.to_numpy()
    if margin:
        if isinstance(margin, (pd.DataFrame, pd.Series)):
            margin = margin.to_numpy()
        margin = margin * f
    else:
        margin = 0 * f
    _data = data[['high', 'low']].to_numpy()

    output = _swing(_data, f, margin)

    if output_as_tuple:
        return data.join(pd.DataFrame(
            output,
            columns=['state', 'extreme', 'pivot', 'signal', 'last_max',
                     'last_min'],
            index=data.index))
    else:
        return output


### Blip to position converter ###

@jit(nopython=True)
def _blip_to_signal_converter(data: np.ndarray) -> np.ndarray:
    """
    Given a column with signals, return a column with positions
    resulting from those signals.

    Args:
    -----

    data: one column with values (-1, 0, 1)

    Returns:
    --------

    Single column array representing positions from input signals.
    """

    state = np.ones(data.shape, dtype=np.int8)

    for i, row in enumerate(data):
        if i == 0:
            state[i] = row
        else:
            # state[i] = np.maximum(np.minimum((state[i-1] + row), 1), -1)
            state[i] = row or state[i-1]

    return state


### In-Out signal unifier ###

@jit(nopython=True)
def _in_out_signal_unifier(data: np.ndarray) -> np.ndarray:
    """
    Given an array with two columns, one with entry and one with close
    signal, return an array with signal resulting from combining those
    two signals.

    Args:
    -----

    data: must have two columns: [0] - entry signal [1] -
    position clos signal

    Returns:
    --------

    Numpy array of shape [data.shape[0], 1] with signal resulting
    from applying entry and close signals (1: long, 0: no position,
    -1: short).
    """

    state = np.zeros((data.shape[0], 1), dtype=np.int8)

    for i, row in enumerate(data):
        if i == 0:
            state[i] = row[0]
        else:
            if state[i-1] != 0:
                state[i] = (np.maximum(np.minimum(
                    (state[i-1] + row[0] + row[1]), 1), -1)).astype(np.int8)
            else:
                state[i] = (np.maximum(np.minimum(
                    (state[i-1] + row[0]), 1), -1)).astype(np.int8)

    return state


## volume grouper ##

@jit(nopython=True)
def _volume_grouper(volume: np.ndarray, target: int) -> np.ndarray:
    aggregator = 0
    label = 0
    labels = np.zeros(volume.shape)
    for i, row in enumerate(volume):
        aggregator += row
        if aggregator >= target:
            aggregator = 0
            label += 1
        labels[i] = label
    return labels


def volume_grouper(df: pd.DataFrame, target) -> pd.DataFrame:
    assert set(['open', 'high', 'low', 'close', 'volume', 'barCount']
               ).issubset(set(df.columns)), (
                   "df must have all of the columns: 'open', 'high', 'low',"
                   " 'close', 'volume', 'barCount'")
    df = df.copy().reset_index()
    df['labels'] = _volume_grouper(df.volume.to_numpy(), target)
    return df.groupby('labels').agg({
        'date': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'barCount': 'sum'
    }).set_index('date')
