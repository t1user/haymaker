from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from haymaker.research import bootstrap, optimal_block_length, prepare_bootstrap_frame


def _ohlc_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            "2024-01-02 09:30",
            "2024-01-02 09:31",
            "2024-01-02 09:32",
            "2024-01-02 09:33",
            "2024-01-03 09:30",
        ],
        name="date",
    )
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 103.0, 102.0, 106.0],
            "high": [102.0, 104.0, 105.0, 107.0, 108.0],
            "low": [99.0, 100.0, 101.0, 101.0, 105.0],
            "close": [101.0, 103.0, 102.0, 106.0, 107.0],
            "volume": [10, 20, 30, 40, 50],
            "barCount": [1, 2, 3, 4, 5],
            "average": [100.5, 102.0, 103.0, 104.0, 106.5],
            "ignored": [1, 1, 1, 1, 1],
        },
        index=index,
    )


def test_prepare_bootstrap_frame_anchors_ohlc_to_previous_close() -> None:
    source = _ohlc_frame()

    actual = prepare_bootstrap_frame(source)

    assert list(actual.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "barCount",
    ]
    assert actual.index.equals(source.index[1:])
    assert actual.loc[source.index[1], "open"] == pytest.approx(
        np.log(
            source.loc[source.index[1], "open"]
            / source.loc[source.index[0], "close"]
        )
    )
    assert actual.loc[source.index[1], "high"] == pytest.approx(
        np.log(
            source.loc[source.index[1], "high"]
            / source.loc[source.index[0], "close"]
        )
    )
    assert actual.loc[source.index[1], "low"] == pytest.approx(
        np.log(
            source.loc[source.index[1], "low"]
            / source.loc[source.index[0], "close"]
        )
    )
    assert actual.loc[source.index[1], "close"] == pytest.approx(
        np.log(
            source.loc[source.index[1], "close"] / source.loc[source.index[0], "close"]
        )
    )
    assert actual.loc[source.index[1], "volume"] == source.loc[
        source.index[1], "volume"
    ]
    assert actual.loc[source.index[1], "barCount"] == source.loc[
        source.index[1], "barCount"
    ]


def test_moving_bootstrap_full_block_reconstructs_source_path() -> None:
    source = _ohlc_frame()

    actual = bootstrap(
        source,
        method="moving",
        block_length=len(source) - 1,
        random_state=0,
    )

    assert isinstance(actual, list)
    assert len(actual) == 1
    pd.testing.assert_index_equal(actual[0].index, source.index[1:])
    pd.testing.assert_frame_equal(
        actual[0], source.iloc[1:].drop(columns=["average", "ignored"])
    )


def test_bootstrap_paths_above_one_returns_list() -> None:
    source = _ohlc_frame()

    actual = bootstrap(source, block_length=2, paths=3, random_state=1)

    assert isinstance(actual, list)
    assert len(actual) == 3
    assert all(isinstance(path, pd.DataFrame) for path in actual)


def test_bootstrap_single_path_returns_list() -> None:
    source = _ohlc_frame()

    actual = bootstrap(source, method="moving", block_length=len(source) - 1)

    assert isinstance(actual, list)
    assert len(actual) == 1
    assert actual[0].index.equals(source.index[1:])
    assert len(actual[0]) == len(source) - 1


def test_bootstrap_preserves_observed_index_after_anchor_row() -> None:
    source = _ohlc_frame()

    actual = bootstrap(source, method="moving", block_length=len(source) - 1)

    pd.testing.assert_index_equal(actual[0].index, source.index[1:])


@pytest.mark.parametrize("method", ["stationary", "moving", "circular"])
def test_bootstrap_supports_all_block_methods(method: str) -> None:
    source = _ohlc_frame()

    actual = bootstrap(
        source,
        method=method,  # type: ignore[arg-type]
        block_length=2,
        random_state=2,
    )

    assert isinstance(actual, list)
    assert len(actual) == 1
    assert len(actual[0]) == len(source) - 1
    assert list(actual[0].columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "barCount",
    ]


def test_explicit_moving_block_length_cannot_exceed_sample_length() -> None:
    source = _ohlc_frame()

    with pytest.raises(ValueError, match="moving block_length"):
        bootstrap(source, method="moving", block_length=len(source))


def test_circular_block_length_can_exceed_sample_length() -> None:
    source = _ohlc_frame()

    actual = bootstrap(
        source, method="circular", block_length=len(source) + 5, random_state=4
    )

    assert len(actual) == 1
    assert len(actual[0]) == len(source) - 1


def test_optimal_block_length_returns_positive_integer() -> None:
    actual = optimal_block_length(pd.Series([0.01, 0.02, -0.01, 0.03, 0.0]))

    assert actual >= 1
