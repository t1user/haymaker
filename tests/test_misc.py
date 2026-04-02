import datetime
from typing import Literal

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker.misc import (
    Counter,
    concat_dfs,
    contractAsTuple,
    format_timestamp,
    general_to_specific_contract_class,
    sign,
)


def test_Counter():
    c = Counter()
    num = c()
    assert isinstance(num, str)
    assert num.endswith("00000")


def test_Counter_increments_by_one():
    c = Counter()
    num = c()
    num1 = c()
    assert int(num1[-1]) - int(num[-1]) == 1


def test_Counter_doesnt_duplicate_on_reinstantiation():
    c = Counter()
    num = c()
    d = Counter()
    num1 = d()
    assert num != num1


@pytest.mark.parametrize(
    "input,expected",
    [(0, 0), (5, 1), (-10, -1), (-3.4, -1), (2.234, 1), (-0, 0), (+0, 0), (-0.0000, 0)],
)
def test_sign_function(input: float | Literal[0] | Literal[5] | Literal[-10], expected):
    assert sign(input) == expected


@pytest.mark.parametrize(
    "contract",
    [
        ibi.Stock,
        ibi.Option,
        ibi.Future,
        ibi.ContFuture,
        ibi.Index,
        ibi.CFD,
        ibi.Bond,
        ibi.Commodity,
        ibi.FuturesOption,
        ibi.MutualFund,
        ibi.Warrant,
        ibi.Crypto,
    ],
)
def test_contractAsTuple_works_for_every_contract_type_except_for_bag(contract):
    tuples = contractAsTuple(contract())
    assert isinstance(tuples, tuple)
    assert isinstance(tuples[0], tuple)


def test_general_to_specific_contract_class():
    contract = ibi.Contract(
        secType="FUT",
        conId=657106382,
        symbol="HSI",
        lastTradeDateOrContractMonth="20240130",
        multiplier="50",
        exchange="HKFE",
        currency="HKD",
        localSymbol="HSIF4",
        tradingClass="HSI",
    )
    future = general_to_specific_contract_class(contract)

    assert future == contract
    assert isinstance(future, ibi.Future)


def test_general_to_specific_contract_class_with_contfuture():
    contract = ibi.ContFuture(
        conId=656780482,
        symbol="MGC",
        lastTradeDateOrContractMonth="20250827",
        multiplier="10",
        exchange="COMEX",
        currency="USD",
        localSymbol="MGCQ5",
        tradingClass="MGC",
    )
    future = general_to_specific_contract_class(contract)

    assert future == contract
    assert isinstance(future, ibi.Future)


def test_general_to_specific_contract_class_with_contfuture_Contract():
    contract = ibi.Contract(
        secType="CONTFUT",
        conId=674701641,
        symbol="MGC",
        lastTradeDateOrContractMonth="20251229",
        multiplier="10",
        exchange="COMEX",
        currency="USD",
        localSymbol="MGCZ5",
        tradingClass="MGC",
    )

    future = general_to_specific_contract_class(contract)

    assert future == contract
    assert isinstance(future, ibi.Future)


def test_general_to_specific_contract_class_works_with_non_futures():
    contract = ibi.Contract(
        secType="STK",
        conId=4391,
        symbol="AMD",
        exchange="SMART",
        primaryExchange="NASDAQ",
        currency="USD",
        localSymbol="AMD",
        tradingClass="NMS",
        comboLegs=[],
    )

    modified = general_to_specific_contract_class(contract)

    assert modified == contract
    assert isinstance(modified, ibi.Stock)


def test_general_to_specific_contract_class_doesnt_touch_contract_subclasses():
    contract = ibi.Future(
        conId=637533641,
        symbol="ES",
        lastTradeDateOrContractMonth="20250919",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESU5",
        tradingClass="ES",
    )
    future = general_to_specific_contract_class(contract)
    assert future is contract


def test_general_to_specific_contract_class_raises_with_non_contracts():
    some_faulty_object = object()
    with pytest.raises(AssertionError):
        general_to_specific_contract_class(some_faulty_object)


@pytest.fixture
def master_df():

    return pd.DataFrame.from_dict(
        {
            "open": {
                datetime.date(2025, 12, 5): 6913.5,
                datetime.date(2025, 12, 12): 6942.75,
                datetime.date(2025, 12, 19): 6890.0,
                datetime.date(2025, 12, 26): 6906.25,
                datetime.date(2026, 1, 2): 6980.5,
                datetime.date(2026, 1, 9): 6911.5,
                datetime.date(2026, 1, 16): 7007.0,
                datetime.date(2026, 1, 23): 6918.25,
                datetime.date(2026, 1, 29): 6904.0,
            },
            "high": {
                datetime.date(2025, 12, 5): 6963.0,
                datetime.date(2025, 12, 12): 6988.0,
                datetime.date(2025, 12, 19): 6932.25,
                datetime.date(2025, 12, 26): 6994.0,
                datetime.date(2026, 1, 2): 6984.75,
                datetime.date(2026, 1, 9): 7017.5,
                datetime.date(2026, 1, 16): 7036.25,
                datetime.date(2026, 1, 23): 6969.0,
                datetime.date(2026, 1, 29): 7043.0,
            },
            "low": {
                datetime.date(2025, 12, 5): 6862.0,
                datetime.date(2025, 12, 12): 6864.0,
                datetime.date(2025, 12, 19): 6771.5,
                datetime.date(2025, 12, 26): 6900.5,
                datetime.date(2026, 1, 2): 6866.75,
                datetime.date(2026, 1, 9): 6899.5,
                datetime.date(2026, 1, 16): 6923.25,
                datetime.date(2026, 1, 23): 6814.5,
                datetime.date(2026, 1, 29): 6879.0,
            },
            "close": {
                datetime.date(2025, 12, 5): 6937.0,
                datetime.date(2025, 12, 12): 6890.5,
                datetime.date(2025, 12, 19): 6887.25,
                datetime.date(2025, 12, 26): 6979.25,
                datetime.date(2026, 1, 2): 6900.5,
                datetime.date(2026, 1, 9): 7005.0,
                datetime.date(2026, 1, 16): 6976.75,
                datetime.date(2026, 1, 23): 6945.75,
                datetime.date(2026, 1, 29): 7017.5,
            },
            "volume": {
                datetime.date(2025, 12, 5): 12969.0,
                datetime.date(2025, 12, 12): 283446.0,
                datetime.date(2025, 12, 19): 7452031.0,
                datetime.date(2025, 12, 26): 2766966.0,
                datetime.date(2026, 1, 2): 3982529.0,
                datetime.date(2026, 1, 9): 6045420.0,
                datetime.date(2026, 1, 16): 6435822.0,
                datetime.date(2026, 1, 23): 6888178.0,
                datetime.date(2026, 1, 29): 3233976.0,
            },
            "average": {
                datetime.date(2025, 12, 5): 6913.675,
                datetime.date(2025, 12, 12): 6914.15,
                datetime.date(2025, 12, 19): 6849.1,
                datetime.date(2025, 12, 26): 6949.675,
                datetime.date(2026, 1, 2): 6924.45,
                datetime.date(2026, 1, 9): 6969.475,
                datetime.date(2026, 1, 16): 6983.6,
                datetime.date(2026, 1, 23): 6900.8,
                datetime.date(2026, 1, 29): 6998.325,
            },
            "barCount": {
                datetime.date(2025, 12, 5): 8422,
                datetime.date(2025, 12, 12): 182334,
                datetime.date(2025, 12, 19): 2573069,
                datetime.date(2025, 12, 26): 835993,
                datetime.date(2026, 1, 2): 1181006,
                datetime.date(2026, 1, 9): 1809844,
                datetime.date(2026, 1, 16): 2047671,
                datetime.date(2026, 1, 23): 2375708,
                datetime.date(2026, 1, 29): 1075470,
            },
        }
    )


def test_concat_dfs_1(master_df):
    # dfs have overlapping values
    df1 = master_df.iloc[:5]
    df2 = master_df.iloc[3:]

    cleaned_df = concat_dfs(df1, df2)

    assert len(cleaned_df) == len(master_df)
    assert cleaned_df.equals(master_df)


def test_concat_dfs_2(master_df):
    # dfs are the same
    df1 = master_df
    df2 = master_df

    cleaned_df = concat_dfs(df1, df2)

    assert cleaned_df.equals(master_df)


def test_concat_dfs_3(master_df):
    # dfs don't have overlapping values
    df1 = master_df.iloc[:5]
    df2 = master_df[5:]

    cleaned_df = concat_dfs(df1, df2)

    assert cleaned_df.equals(master_df)


def test_concat_dfs_4(master_df):
    # dfs are not continuous
    df1 = master_df.iloc[:4]
    df2 = master_df[5:]

    cleaned_df = concat_dfs(df1, df2)
    modified_df = master_df.drop(master_df.index[4])
    assert cleaned_df.equals(modified_df)


@pytest.mark.parametrize(
    "dt_input, expected",
    [
        # 1. Native datetime inputs
        pytest.param(
            datetime.datetime(2026, 1, 1, 10, 30),
            datetime.datetime(2026, 1, 1, 10, 30, tzinfo=datetime.timezone.utc),
            id="naive_datetime_to_utc",
        ),
        pytest.param(
            datetime.datetime(2026, 1, 1, 10, 30, tzinfo=datetime.timezone.utc),
            datetime.datetime(2026, 1, 1, 10, 30, tzinfo=datetime.timezone.utc),
            id="preserve_existing_utc_datetime",
        ),
        # 2. Compact YYYYMMDD string format
        pytest.param(
            "20260101",
            datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
            id="compact_date_only_to_utc",
        ),
        # 3. YYYYMMDD HH:MM:SS TZ format (Interactive Brokers style)
        pytest.param(
            "20260101 15:30:00 UTC",
            datetime.datetime(2026, 1, 1, 15, 30, tzinfo=datetime.timezone.utc),
            id="compact_with_utc_tz",
        ),
        pytest.param(
            "20260101 15:30:00 Z",
            datetime.datetime(2026, 1, 1, 15, 30, tzinfo=datetime.timezone.utc),
            id="compact_with_z_marker",
        ),
        pytest.param(
            "20260101 15:30:00",
            datetime.datetime(2026, 1, 1, 15, 30, tzinfo=datetime.timezone.utc),
            id="compact_no_tz_fallback_to_utc",
        ),
        # 4. Standard ISO formats (Handled by fromisoformat)
        pytest.param(
            "2026-01-01",
            datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
            id="iso_date_only_to_utc",
        ),
        pytest.param(
            "2026-01-01T15:30:00Z",
            datetime.datetime(2026, 1, 1, 15, 30, tzinfo=datetime.timezone.utc),
            id="iso_with_zulu_marker",
        ),
        pytest.param(
            "2026-01-01 15:30:00",
            datetime.datetime(2026, 1, 1, 15, 30, tzinfo=datetime.timezone.utc),
            id="iso_no_tz_fallback_to_utc",
        ),
    ],
)
def test_format_timestamp_valid_inputs(dt_input, expected):
    """Verifies that various valid date formats are parsed accurately and fall back to UTC."""
    result = format_timestamp(dt_input)
    assert result == expected
    assert result.tzinfo == expected.tzinfo


def test_format_timestamp_preserves_foreign_timezone():
    """Verifies that a string holding an offset other than UTC is preserved."""
    # Custom timezone offset (+02:00)
    iso_with_offset = "2026-01-01T15:30:00+02:00"

    result = format_timestamp(iso_with_offset)

    assert result.tzinfo is not None
    # Verifies the offset remains exactly +02:00 (7200 seconds)
    assert result.utcoffset().total_seconds() == 7200


@pytest.mark.parametrize("invalid_input", [12345678, ["20260101"], None])
def test_format_timestamp_type_error(invalid_input):
    """Verifies that passing an unsupported type triggers a TypeError."""
    with pytest.raises(TypeError):
        format_timestamp(invalid_input)
