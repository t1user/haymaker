from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from haymaker.research import (
    bootstrap,
    combine_states,
    hmm_states,
    optimal_block_length,
    prepare_bootstrap_frame,
    range_states,
    regime_bootstrap,
    return_states,
    trend_states,
    volatility_states,
)
from haymaker.research.bootstrap import regime_bootstrap as package_regime_bootstrap


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


def _regime_ohlc_frame(rows: int = 121) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 09:30", periods=rows, freq="min", name="date")
    close = pd.Series(100 * np.exp(np.linspace(0, 0.2, rows)), index=index)
    open_ = close.shift(fill_value=close.iloc[0]) * 1.0005
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.002
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.998
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.arange(rows) + 100,
            "barCount": np.arange(rows) + 1,
        },
        index=index,
    )


def _alternating_states(index: pd.Index, labels: tuple[object, object]) -> pd.Series:
    values = [labels[location % 2] for location in range(len(index))]
    return pd.Series(values, index=index, name="state")


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


def test_bootstrap_public_imports_work() -> None:
    assert bootstrap is not None
    assert regime_bootstrap is package_regime_bootstrap


@pytest.mark.parametrize(
    "labels",
    [
        ("quiet", "volatile"),
        (0, 1),
        (("up", "low_vol"), ("down", "high_vol")),
    ],
)
def test_regime_bootstrap_accepts_hashable_state_labels(
    labels: tuple[object, object],
) -> None:
    source = _regime_ohlc_frame()
    states = _alternating_states(source.index, labels)

    actual = regime_bootstrap(source, states=states, paths=2, random_state=1)

    assert isinstance(actual, list)
    assert len(actual) == 2
    for path in actual:
        pd.testing.assert_index_equal(path.index, source.index[1:])
        assert len(path) == len(source) - 1
        assert list(path.columns) == [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "barCount",
        ]


def test_regime_bootstrap_rejects_missing_state_labels() -> None:
    source = _regime_ohlc_frame()
    states = _alternating_states(source.index, ("a", "b"))
    states.iloc[10] = np.nan

    with pytest.raises(ValueError, match="missing state labels"):
        regime_bootstrap(
            source,
            states=states,
            min_state_count=1,
            min_transition_count=1,
        )


def test_regime_bootstrap_rejects_too_few_rows_per_state() -> None:
    source = _regime_ohlc_frame()
    states = pd.Series("common", index=source.index, name="state")
    states.iloc[-1] = "rare"

    with pytest.raises(ValueError, match="too few rows"):
        regime_bootstrap(
            source,
            states=states,
            min_state_count=2,
            min_transition_count=1,
        )


def test_regime_bootstrap_rejects_too_few_outgoing_transitions() -> None:
    source = _regime_ohlc_frame()
    midpoint = len(source) // 2
    states = pd.Series("a", index=source.index, name="state")
    states.iloc[midpoint:] = "b"

    with pytest.raises(ValueError, match="too few transitions"):
        regime_bootstrap(
            source,
            states=states,
            min_state_count=30,
            min_transition_count=70,
        )


def test_combine_states_returns_tuple_labels_on_common_non_null_index() -> None:
    index = pd.Index([1, 2, 3, 4])
    first = pd.Series(["up", "up", None, "down"], index=index)
    second = pd.Series(["low", "high", "low"], index=pd.Index([2, 3, 4]))

    actual = combine_states(first, second)

    expected = pd.Series([("up", "low"), ("down", "low")], index=pd.Index([2, 4]))
    pd.testing.assert_series_equal(actual, expected.rename("state"))


def test_state_helpers_return_feedable_state_series() -> None:
    source = _regime_ohlc_frame()

    states = combine_states(
        trend_states(source, window=3),
        volatility_states(source, window=3),
        range_states(source),
        return_states(source),
    )
    actual = regime_bootstrap(
        source,
        states=states,
        paths=1,
        random_state=2,
        min_state_count=1,
        min_transition_count=1,
    )

    assert len(actual) == 1
    pd.testing.assert_index_equal(actual[0].index, source.index[1:])


def test_hmm_states_return_feedable_state_series() -> None:
    pytest.importorskip("hmmlearn")
    source = _regime_ohlc_frame()

    states = hmm_states(source, n_states=2, random_state=1)
    actual = regime_bootstrap(
        source,
        states=states,
        paths=1,
        random_state=2,
        min_state_count=1,
        min_transition_count=1,
    )

    assert states.index.equals(source.index[1:])
    assert states.nunique() <= 2
    assert len(actual) == 1
    pd.testing.assert_index_equal(actual[0].index, source.index[1:])


def test_hmm_states_rejects_too_few_states() -> None:
    source = _regime_ohlc_frame()

    with pytest.raises(ValueError, match="n_states"):
        hmm_states(source, n_states=1)
