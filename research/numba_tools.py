import pandas as pd
import numpy as np
from numba import jit
from typing import Optional, Union, NamedTuple


### Swing trading signals ###

@jit(nopython=True)
def _swing(data: np.array, f: np.array, margin: np.array) -> np.array:
    """Numba optimized implementation for swing indicators.  Should
    not be used directly, but via swing function which accepts pandas
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
                max_list[i] = max_pivot
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
                min_list[i] = min_pivot
            else:
                state[i, 0] = -1
                extreme[i] = min(extreme[i-1, 0], row[1])
                pivot[i] = pivot[i-1]

        else:
            raise ValueError('Wrong state value!')
    return np.concatenate((state, extreme, pivot, signal, max_list, min_list),
                          axis=1)


class Output(NamedTuple):
    state: np.array
    extreme: np.array
    pivot: np.array
    signal: np.array
    last_max: np.array
    last_min: np.array


def _output_converter(arr: np.array) -> Output:
    return Output(*arr.T)


def swing(data: pd.DataFrame, f: Union[float, np.array, pd.Series],
          margin: Optional[float] = None, output_as_tuple: bool = True
          ) -> Union[np.array, Output]:
    """
    Simulate swing trading signals and return generated output series.

    User interface allowing to use numba optimized _swing function
    passing pandas objects (See _swing docstring).  This should be
    used as entry point to _swing.

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

    Numpy array with shape [data.shape[0], 6] (if output_as_tuple is
    False) or named tuple with pd.Series, where columns/properties
    have the following meaning for every row corresponding to price
    data ([np.array column index]/NamedTuple property name):

    [0]/state - what swing are we in?  (1) for upswing (-1) for
    downswing

    [1]/extreme - current extreme (since last pivot) against which
    filter is applied to determine if swing changed direction

    [2]/pivot - last swing pivot point

    [3]/signal - trade signal, generated when opposite swing pivot is
    breached by margin

    [4]/last_max - at pivot point latest upswing pivot, otherwise zero

    [5]/last_min - at pivot point latest downswing pivot, otherwize zero
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
    data = data[['high', 'low']].to_numpy()

    output = _swing(data, f, margin)

    if output_as_tuple:
        return _output_converter(output)
    else:
        return output


### Signal to position converter ###

@jit(nopython=True)
def _signal_to_position_converter(data: np.array) -> np.array:
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


### In-Out signal to position converter ###

@jit(nopython=True)
def _in_out_signal_to_position_converter(data: np.array) -> np.array:
    """
    Given an array with two columns, one with position entry signals
    and one with position close signals, return an array with
    corresponding position after applying those two signals.

    Args:
    -----

    data - must have two columns: [0] - position entry signals [1] -
    position close signals

    Returns:
    --------

    Numpy array of shape [data.shape[0], 1] with position resulting
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
