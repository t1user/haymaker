from __future__ import annotations

import pandas as pd
import pytest

from haymaker.research import (
    excursions,
    factor_extractor,
    winning_trade_adverse_excursions,
)
from haymaker.research.backtester import Results


def _market_data() -> pd.DataFrame:
    """Return deterministic bars with an extreme exit bar."""
    index = pd.date_range("2026-01-01 09:00", periods=3, freq="h")
    return pd.DataFrame(
        {
            "high": [105.0, 110.0, 120.0],
            "low": [98.0, 95.0, 80.0],
        },
        index=index,
    )


def _position(
    open_price: float,
    close_price: float,
    gross_pnl: float,
    *,
    same_bar: bool = False,
) -> pd.DataFrame:
    """Return one completed trade using the backtester position schema."""
    index = _market_data().index
    return pd.DataFrame(
        {
            "date_o": [index[0]],
            "open": [open_price],
            "date_c": [index[0] if same_bar else index[2]],
            "close": [close_price],
            "g_pnl": [gross_pnl],
            "pnl": [gross_pnl],
        },
        index=pd.Index([7], name="trade_id"),
    )


@pytest.mark.parametrize(
    ("open_price", "close_price", "gross_pnl", "expected"),
    [
        (100.0, -103.0, 3.0, {"fav": 10.0, "adv": 5.0, "eff": 0.2}),
        (-100.0, 96.0, 4.0, {"fav": 5.0, "adv": 10.0, "eff": 0.27}),
    ],
)
def test_excursions_uses_direction_and_excludes_exit_bar(
    open_price: float,
    close_price: float,
    gross_pnl: float,
    expected: dict[str, float],
) -> None:
    actual = excursions(
        _market_data(),
        _position(open_price, close_price, gross_pnl),
    )

    expected_frame = pd.DataFrame(
        [expected],
        index=pd.Index([7], name="trade_id"),
    )
    pd.testing.assert_frame_equal(actual, expected_frame)


def test_excursions_same_bar_trade_uses_only_execution_prices() -> None:
    positions = _position(100.0, -95.0, -5.0, same_bar=True)

    actual = excursions(_market_data(), positions)

    expected = pd.DataFrame(
        [{"fav": 0.0, "adv": 5.0, "eff": -1.0}],
        index=pd.Index([7], name="trade_id"),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_excursions_scales_at_entry_and_preserves_position_index() -> None:
    market = _market_data()
    divisor = pd.Series([2.0, 4.0, 8.0], index=market.index)

    actual = excursions(market, _position(100.0, -103.0, 3.0), divisor)

    expected = pd.DataFrame(
        [{"pnl_mul": 1.5, "fav": 5.0, "adv": 2.5, "eff": 0.2}],
        index=pd.Index([7], name="trade_id"),
    )
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda market, positions: market.drop(columns="low"), "missing required"),
        (
            lambda market, positions: market.set_axis(
                [market.index[1], market.index[0], market.index[2]]
            ),
            "sorted",
        ),
        (
            lambda market, positions: market.assign(low=market["high"] + 1),
            "below its low",
        ),
    ],
)
def test_excursions_rejects_invalid_market_data(mutation, message: str) -> None:
    market = _market_data()
    positions = _position(100.0, -103.0, 3.0)

    with pytest.raises(ValueError, match=message):
        excursions(mutation(market, positions), positions)


def test_excursions_rejects_misaligned_or_non_positive_divisor() -> None:
    market = _market_data()
    positions = _position(100.0, -103.0, 3.0)

    with pytest.raises(ValueError, match="exactly the same index"):
        excursions(market, positions, pd.Series([1.0]))

    divisor = pd.Series([1.0, 0.0, 1.0], index=market.index)
    with pytest.raises(ValueError, match="positive"):
        excursions(market, positions, divisor)


def test_winning_trade_adverse_excursions_filters_on_net_pnl() -> None:
    market = _market_data()
    positions = pd.concat(
        [
            _position(100.0, -103.0, 3.0),
            _position(-100.0, 96.0, 4.0).set_axis([8]),
        ]
    )
    positions.loc[8, "pnl"] = -1.0
    results = Results(
        stats=pd.Series(dtype=float),
        daily=pd.DataFrame(),
        positions=positions,
        df=pd.DataFrame(),
        warnings=[],
    )

    full = winning_trade_adverse_excursions(market, results, full=True)
    description = winning_trade_adverse_excursions(market, results)

    assert full.index.tolist() == [7]
    assert full.loc[7, "adv"] == 5.0
    assert description["count"] == 1.0
    assert description["mean"] == 5.0


def test_factor_extractor_shifts_and_preserves_duplicate_entry_rows() -> None:
    index = pd.date_range("2026-01-01 09:00", periods=3, freq="h")
    positions = pd.DataFrame(
        {
            "date_o": [index[1], index[1]],
            "pnl": [1.0, 2.0],
            "factor": [-1.0, -1.0],
        },
        index=[10, 20],
    )
    data = pd.DataFrame(
        {"factor": [10.0, 20.0, 30.0], "atr": [1.0, 2.0, 3.0]},
        index=index,
    )

    actual = factor_extractor(positions, data, ["factor", "atr"])

    expected = pd.DataFrame(
        {
            "date_o": [index[1], index[1]],
            "pnl": [1.0, 2.0],
            "factor": [10.0, 10.0],
            "atr": [1.0, 1.0],
        }
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_factor_extractor_can_use_values_available_at_entry() -> None:
    index = pd.date_range("2026-01-01 09:00", periods=2, freq="h")
    positions = pd.DataFrame({"date_o": [index[1]]})
    data = pd.DataFrame({"factor": [10.0, 20.0]}, index=index)

    actual = factor_extractor(positions, data, "factor", shift=False)

    assert actual.loc[0, "factor"] == 20.0


@pytest.mark.parametrize(
    ("positions", "data", "field", "message"),
    [
        (pd.DataFrame({"pnl": [1.0]}), pd.DataFrame(), "factor", "positions"),
        (
            pd.DataFrame({"date_o": [pd.Timestamp("2026-01-01")]}),
            pd.DataFrame(index=[pd.Timestamp("2026-01-01")]),
            "factor",
            "data",
        ),
        (
            pd.DataFrame({"date_o": [pd.Timestamp("2026-01-02")]}),
            pd.DataFrame({"factor": [1.0]}, index=[pd.Timestamp("2026-01-01")]),
            "factor",
            "entry dates",
        ),
    ],
)
def test_factor_extractor_rejects_invalid_inputs(
    positions: pd.DataFrame,
    data: pd.DataFrame,
    field: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factor_extractor(positions, data, field)


def test_excursions_empty_positions_preserves_schema() -> None:
    positions = pd.DataFrame(
        columns=["date_o", "open", "date_c", "close", "g_pnl", "pnl"],
        index=pd.Index([], name="trade_id"),
    )

    actual = excursions(_market_data(), positions)

    expected = pd.DataFrame(
        columns=["fav", "adv", "eff"],
        index=pd.Index([], name="trade_id"),
        dtype=float,
    )
    pd.testing.assert_frame_equal(actual, expected)
