from __future__ import annotations

from itertools import product
from typing import Callable, NotRequired, TypedDict

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from haymaker.research.signal_converters import (
    pos_trans,
    pos_trans_array,
    pos_trans_numpy,
)
from haymaker.research.stop import StopMode, stop_loss
from haymaker.research.stop.interface import PreparedData, _prepare_data
from haymaker.research.stop.numba_impl import run_stop_loss as run_stop_loss_numba
from haymaker.research.stop.python_impl import (
    StopParams,
    param_factory,
    run_stop_loss as run_stop_loss_python,
)


class StopKwargs(TypedDict):
    mode: StopMode
    tp_multiple: NotRequired[float]
    adjust: NotRequired[tuple[StopMode, float, float]]
    time_stop: NotRequired[int]


def _run_engine(
    prepared: PreparedData, params: StopParams, use_numba: bool
) -> pd.DataFrame:
    if use_numba:
        result = run_stop_loss_numba(
            prepared.first,
            prepared.second,
            prepared.high,
            prepared.low,
            prepared.distance,
            prepared.price,
            prepared.use_blip,
            params,
        )
    else:
        result = run_stop_loss_python(
            prepared.first,
            prepared.second,
            prepared.high,
            prepared.low,
            prepared.distance,
            prepared.price,
            prepared.use_blip,
            params,
        )

    out = pd.DataFrame(
        result,
        columns=["position", "open_price", "close_price", "stop_price"],
        index=prepared.row_index,
    )
    out["position"] = out["position"].astype(int, copy=False)
    return out


def _price_frame(length: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    open_price = 100 + np.cumsum(rng.normal(0, 1.2, size=length))
    span_high = rng.uniform(0.1, 2.5, size=length)
    span_low = rng.uniform(0.1, 2.5, size=length)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": open_price + span_high,
            "low": open_price - span_low,
        }
    )


def _position_frame(length: int, seed: int) -> pd.DataFrame:
    df = _price_frame(length, seed)
    rng = np.random.default_rng(seed + 101)
    states = np.array([-1, 0, 1], dtype=int)
    df["position"] = rng.choice(states, size=length, p=[0.3, 0.35, 0.35])
    df.loc[0, "position"] = 0
    return df


def _blip_frame(length: int, seed: int) -> pd.DataFrame:
    df = _price_frame(length, seed)
    rng = np.random.default_rng(seed + 202)
    states = np.array([-1, 0, 1], dtype=int)
    position = pd.Series(
        rng.choice(states, size=length, p=[0.25, 0.45, 0.3]), index=df.index
    )
    position.iloc[0] = 0
    previous = position.shift(fill_value=0)
    df["blip"] = position.where((position != 0) & (position != previous), 0).astype(int)
    df["close_blip"] = (
        (-previous).where((previous != 0) & (position != previous), 0).astype(int)
    )
    return df


def _distance_options(df: pd.DataFrame) -> list[float | pd.Series]:
    scalar = 1.75
    series = pd.Series(
        np.linspace(0.8, 2.4, len(df)),
        index=df.index,
        dtype=float,
    )
    return [scalar, series]


def test_stop_loss_rejects_distance_series_with_mismatched_index() -> None:
    df = _position_frame(10, seed=101)
    wrong_index = pd.RangeIndex(1, len(df) + 1)
    distance = pd.Series(1.75, index=wrong_index)

    with pytest.raises(ValueError, match="same index as df"):
        stop_loss(df, distance)


def _stop_kwargs() -> list[StopKwargs]:
    return [
        {"mode": "fixed"},
        {"mode": "trail"},
        {"mode": "fixed", "tp_multiple": 3.0},
        {"mode": "trail", "tp_multiple": 3.0},
        {"mode": "fixed", "time_stop": 2},
        {"mode": "trail", "time_stop": 3},
        {"mode": "fixed", "adjust": ("fixed", 1.5, 1.0)},
        {"mode": "trail", "adjust": ("fixed", 1.5, 1.0)},
        {"mode": "fixed", "adjust": ("trail", 1.5, 1.0)},
        {"mode": "trail", "adjust": ("trail", 1.5, 1.0)},
        {
            "mode": "fixed",
            "tp_multiple": 3.0,
            "time_stop": 2,
            "adjust": ("fixed", 1.5, 1.0),
        },
        {
            "mode": "trail",
            "tp_multiple": 3.0,
            "time_stop": 2,
            "adjust": ("trail", 1.5, 1.0),
        },
    ]


@pytest.mark.parametrize(
    ("df", "distance", "kwargs"),
    [
        (
            pd.DataFrame(
                {
                    "open": [100.0, 100.0, 101.0],
                    "high": [100.0, 101.0, 101.0],
                    "low": [100.0, 97.0, 100.0],
                    "position": [0, 1, 0],
                }
            ),
            2.0,
            {"mode": "fixed"},
        ),
        (
            pd.DataFrame(
                {
                    "open": [100.0, 100.0, 101.0, 103.0, 102.0, 100.0],
                    "high": [100.0, 102.0, 104.0, 105.0, 103.0, 101.0],
                    "low": [100.0, 99.0, 101.0, 102.0, 100.0, 99.0],
                    "position": [0, 1, 1, 1, 1, 0],
                }
            ),
            2.0,
            {"mode": "trail", "adjust": ("fixed", 2.0, 1.0)},
        ),
        (
            pd.DataFrame(
                {
                    "open": [100.0, 100.0, 101.0, 102.0, 103.0],
                    "high": [100.0, 102.0, 104.0, 105.0, 103.0],
                    "low": [100.0, 99.0, 100.0, 101.0, 102.0],
                    "position": [0, 1, 1, 1, 0],
                }
            ),
            1.0,
            {"mode": "fixed", "tp_multiple": 3.0},
        ),
        (
            pd.DataFrame(
                {
                    "open": [100.0, 100.0, 101.0, 102.0, 103.0],
                    "high": [100.0, 101.0, 102.0, 103.0, 104.0],
                    "low": [100.0, 99.0, 100.0, 101.0, 102.0],
                    "position": [0, 1, 1, 1, 0],
                }
            ),
            10.0,
            {"mode": "fixed", "time_stop": 2},
        ),
        (
            pd.DataFrame(
                {
                    "open": [100.0, 101.0, 102.0, 103.0, 101.0],
                    "high": [101.0, 102.0, 103.0, 104.0, 102.0],
                    "low": [99.0, 100.0, 101.0, 102.0, 100.0],
                    "blip": [1, 0, 0, -1, 0],
                    "close_blip": [0, 0, -1, 0, 0],
                }
            ),
            2.0,
            {"mode": "trail"},
        ),
    ],
)
def test_stop_loss_matches_between_engines_on_targeted_cases(
    df: pd.DataFrame, distance: float, kwargs: StopKwargs
) -> None:
    expected = stop_loss(df, distance, **kwargs, use_numba=False)
    actual = stop_loss(df, distance, **kwargs, use_numba=True)

    pdt.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        atol=1e-7,
        rtol=0,
    )


@pytest.mark.parametrize(("length", "seed"), [(25, 7), (40, 19)])
def test_position_transaction_helpers_match(length: int, seed: int) -> None:
    df = _position_frame(length, seed)

    pandas_series = pos_trans(df["position"])
    numpy_series = pos_trans_numpy(df["position"])
    numpy_array = pos_trans_array(df["position"].to_numpy(dtype=np.int8, copy=False))

    pdt.assert_series_equal(numpy_series, pandas_series, check_dtype=False)
    np.testing.assert_array_equal(
        numpy_array, pandas_series.to_numpy(dtype=np.int8, copy=False)
    )


@pytest.mark.parametrize(
    ("frame_factory", "length", "seed"),
    [(_position_frame, 25, 7), (_position_frame, 40, 19)],
)
def test_transaction_prep_methods_match_for_stop_loss(
    frame_factory: Callable[[int, int], pd.DataFrame], length: int, seed: int
) -> None:
    df = frame_factory(length, seed)
    params = param_factory("trail", 3.0, 2, ("fixed", 1.5, 1.0))

    prepared_numpy = _prepare_data(df, 1.75, "open", transaction_method="numpy")
    prepared_pandas = _prepare_data(df, 1.75, "open", transaction_method="pandas")

    np.testing.assert_array_equal(prepared_numpy.first, prepared_pandas.first)
    np.testing.assert_array_equal(prepared_numpy.second, prepared_pandas.second)

    python_numpy = _run_engine(prepared_numpy, params, use_numba=False)
    python_pandas = _run_engine(prepared_pandas, params, use_numba=False)
    numba_numpy = _run_engine(prepared_numpy, params, use_numba=True)
    numba_pandas = _run_engine(prepared_pandas, params, use_numba=True)

    pdt.assert_frame_equal(python_numpy, python_pandas, check_dtype=False)
    pdt.assert_frame_equal(numba_numpy, numba_pandas, check_dtype=False)


@pytest.mark.parametrize(
    ("frame_factory", "length", "seed"),
    [
        (_position_frame, 25, 7),
        (_position_frame, 40, 19),
        (_blip_frame, 25, 11),
        (_blip_frame, 40, 23),
    ],
)
def test_stop_loss_matches_between_engines_across_parameter_grid(
    frame_factory: Callable[[int, int], pd.DataFrame], length: int, seed: int
) -> None:
    df = frame_factory(length, seed)

    for distance, kwargs in product(_distance_options(df), _stop_kwargs()):
        expected = stop_loss(df, distance, **kwargs, use_numba=False)
        actual = stop_loss(df, distance, **kwargs, use_numba=True)
        pdt.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            atol=1e-7,
            rtol=0,
            obj=f"seed={seed}, kwargs={kwargs}, use_blip={'blip' in df.columns}",
        )
