import math
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import haymaker.research.grid_search as grid_search
from haymaker.research.grid_search import (
    GridSearch,
    GridSearchResult,
    JobKey,
    combined_path,
    combined_returns,
    combined_stats,
    plot_grid,
    show_grid,
    show_grid_table,
)


def test_public_api_is_explicit() -> None:
    assert set(grid_search.__all__) == {
        "GridSearch",
        "GridSearchResult",
        "combined_returns",
        "combined_path",
        "combined_stats",
        "plot_grid",
        "show_grid",
        "show_grid_table",
    }


def _price_df(offset: float = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [100 + offset, 101 + offset, 102 + offset, 103 + offset]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )


def _strategy(
    close: pd.Series,
    first: float = 0,
    second: float = 0,
    *,
    bias: float = 0,
) -> pd.DataFrame:
    from haymaker.research.backtester import no_stop

    threshold = close.iloc[0] + float(first) - float(second) + bias
    strategy = pd.DataFrame({"close": close})
    strategy["position"] = (close > threshold).astype(int)
    return no_stop(strategy, price_column="close")


def _full_df_strategy(
    df: pd.DataFrame,
    first: float = 0,
    second: float = 0,
    *,
    bias: float = 0,
) -> pd.DataFrame:
    return _strategy(df["close"], first, second, bias=bias)


def _result_with_returns() -> GridSearchResult:
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    raw_stats: dict[JobKey, pd.Series] = {
        ("a", 1): pd.Series({"Annual return": 0.1, "Sharpe ratio": 1.0}),
        ("b", 1): pd.Series({"Annual return": 0.2, "Sharpe ratio": 2.0}),
    }
    raw_dailys: dict[JobKey, pd.DataFrame] = {
        ("a", 1): pd.DataFrame(
            {
                "returns": [0.1, math.nan, 0.2],
                "lreturn": [0.095, math.nan, 0.182],
                "balance": [1.1, math.nan, 1.32],
            },
            index=index,
        ),
        ("b", 1): pd.DataFrame(
            {
                "returns": [0.0, 0.1, 0.0],
                "lreturn": [0.0, 0.095, 0.0],
                "balance": [1.0, 1.1, 1.1],
            },
            index=index,
        ),
    }
    return GridSearchResult(
        raw_stats=raw_stats,
        raw_dailys=raw_dailys,
        raw_positions={key: pd.DataFrame() for key in raw_stats},
        raw_dfs={key: pd.DataFrame() for key in raw_stats},
        raw_warnings={key: [] for key in raw_stats},
    )


def _plot_result() -> GridSearchResult:
    raw_stats: dict[JobKey, pd.Series] = {}
    for first in range(10):
        for second in range(10):
            raw_stats[(first, second)] = pd.Series(
                {
                    "Annual return": (first - second) / 100,
                    "Sharpe ratio": (first + second) / 10,
                }
            )
    return GridSearchResult(
        raw_stats=raw_stats,
        raw_dailys={},
        raw_positions={key: pd.DataFrame() for key in raw_stats},
        raw_dfs={},
        raw_warnings={key: [] for key in raw_stats},
    )


def test_from_progressions_generates_pair_simulations() -> None:
    search = GridSearch.from_progressions(
        _price_df(),
        _strategy,
        [(1, 1, "lin"), (10, 10, "lin")],
        multiprocess=False,
    )

    assert len(search.simulations) == 100
    assert search.simulations[0].key == (1, 10)
    assert search.simulations[0].args == (1, 10)
    assert search.simulations[1].key == (1, 20)


def test_from_pairs_routes_named_params_and_fixed_kwargs() -> None:
    search = GridSearch.from_pairs(
        _price_df(),
        _strategy,
        [(1, 2)],
        param_names=("first", "second"),
        fixed_kwargs={"bias": 3},
        multiprocess=False,
    )

    simulation = search.simulations[0]
    assert simulation.args == ()
    assert simulation.kwargs == {"bias": 3, "first": 1, "second": 2}


def test_param_names_cannot_overlap_fixed_kwargs() -> None:
    with pytest.raises(ValueError, match="overlap"):
        GridSearch.from_pairs(
            _price_df(),
            _strategy,
            [(1, 2)],
            param_names=("first", "second"),
            fixed_kwargs={"first": 3},
        )


def test_from_dfs_supports_mapping_and_sequence_labels() -> None:
    mapping_search = GridSearch.from_dfs(
        {"left": _price_df(), "right": _price_df(10)},
        _strategy,
        params=(1, 2),
        param_names=("first", "second"),
        multiprocess=False,
    )
    sequence_search = GridSearch.from_dfs(
        [_price_df(), _price_df(10)],
        _strategy,
        params=(1, 2),
        multiprocess=False,
    )

    assert [simulation.key for simulation in mapping_search.simulations] == [
        ("left", (1, 2)),
        ("right", (1, 2)),
    ]
    assert [simulation.key for simulation in sequence_search.simulations] == [
        (0, (1, 2)),
        (1, (1, 2)),
    ]


def test_grid_search_run_returns_result_with_tables_and_dynamic_fields() -> None:
    search = GridSearch.from_pairs(
        _price_df(),
        _strategy,
        [(0, 1), (1, 1)],
        multiprocess=False,
    )

    result = search.run()

    assert isinstance(result, GridSearchResult)
    assert "annual_return" in result.fields
    pd.testing.assert_frame_equal(result.tables["annual_return"], result.annual_return)
    assert not result.returns.empty
    with pytest.raises(AttributeError):
        result.not_a_field


def test_stats_frame_shows_all_perf_stats_by_simulation() -> None:
    result = _result_with_returns()

    frame = result.stats_frame

    assert list(frame.index) == ["Annual return", "Sharpe ratio"]
    assert list(frame.columns) == [("a", 1), ("b", 1)]
    assert frame.iloc[0, 0] == 0.1
    assert frame.iloc[0, 1] == 0.2
    assert frame.iloc[1, 0] == 1.0
    assert frame.iloc[1, 1] == 2.0
    assert result.stats_frame is frame


def test_stats_frame_uses_dataframe_labels_for_from_dfs() -> None:
    result = GridSearch.from_dfs(
        {"left": _price_df(), "right": _price_df(10)},
        _strategy,
        params=(1, 2),
        param_names=("first", "second"),
        multiprocess=False,
    ).run()

    assert list(result.raw_stats) == [
        ("left", (1, 2)),
        ("right", (1, 2)),
    ]
    assert list(result.stats_frame.columns) == ["left", "right"]
    assert "annual_return" in result.fields


def test_pass_full_df_sends_full_dataframe_to_strategy() -> None:
    result = GridSearch.from_pairs(
        _price_df(),
        _full_df_strategy,
        [(0, 1)],
        pass_full_df=True,
        multiprocess=False,
    ).run()

    assert "annual_return" in result.fields


def test_save_mem_disables_daily_dependent_properties() -> None:
    result = GridSearch.from_pairs(
        _price_df(),
        _strategy,
        [(0, 1)],
        save_mem=True,
        multiprocess=False,
    ).run()

    with pytest.raises(ValueError, match="Daily data is unavailable"):
        _ = result.returns
    with pytest.raises(ValueError, match="Daily data is unavailable"):
        _ = result.rank


def test_combined_returns_missing_policies() -> None:
    result = _result_with_returns()
    keys = [("a", 1), ("b", 1)]

    zero = result.combined_returns(keys)
    drop = result.combined_returns(keys, missing="drop")

    expected_zero = pd.Series(
        [0.05, 0.05, 0.1],
        index=result.returns.index,
    )
    expected_drop = pd.Series(
        [0.05, 0.1, 0.1],
        index=result.returns.index,
    )
    pd.testing.assert_series_equal(zero, expected_zero)
    pd.testing.assert_series_equal(drop, expected_drop)
    with pytest.raises(ValueError, match="missing daily returns"):
        result.combined_returns(keys, missing="raise")


def test_combined_path_and_stats() -> None:
    result = _result_with_returns()
    keys = [("a", 1), ("b", 1)]

    path = result.combined_path(keys)
    pd.testing.assert_series_equal(
        path,
        (result.combined_returns(keys) + 1).cumprod(),
    )
    assert "perf_stats" not in vars(sys.modules["haymaker.research.grid_search"])
    assert isinstance(result.combined_stats(keys), pd.Series)


def test_combined_function_wrappers_delegate_to_result_methods() -> None:
    result = _result_with_returns()
    keys = [("a", 1), ("b", 1)]

    pd.testing.assert_series_equal(
        combined_returns(result, keys),
        result.combined_returns(keys),
    )
    pd.testing.assert_series_equal(
        combined_path(result, keys),
        result.combined_path(keys),
    )
    assert isinstance(combined_stats(result, keys), pd.Series)


def test_plot_grid_accepts_grid_search_result() -> None:
    result = _plot_result()

    fig = plot_grid(result)

    assert fig.number in plt.get_fignums()
    plt.close(fig)


def test_show_grid_closes_displayed_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    import IPython.display

    monkeypatch.setattr(IPython.display, "display", lambda _: None)

    show_grid(_plot_result())

    assert plt.get_fignums() == []


def test_show_grid_table_accepts_grid_search_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display

    displayed: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)

    show_grid_table(_plot_result(), fields=("annual_return", "sharpe_ratio"))

    assert len(displayed) == 2
