from unittest.mock import Mock

import pandas as pd
import pytest

from haymaker.sticher import FuturesSticher


def test_offset():
    sticher = FuturesSticher("XX", Mock())
    assert sticher.offset(5, 7) == 2


def test_offset_negative():
    sticher = FuturesSticher("XX", Mock())
    assert sticher.offset(5, 4) == -1


def test_offset_mul():
    sticher = FuturesSticher("XXX", Mock(), "mul")
    assert sticher.offset(0.5, 1) == 2


def test_offset_None():
    sticher = FuturesSticher("XXX", Mock(), None)
    assert sticher.offset(5, 7) == 0


def test_sticher_doesnt_accept_wrong_adjust_type():
    with pytest.raises(AssertionError):
        FuturesSticher("XX", Mock(), "xxx")


def test_adjust():
    sticher = FuturesSticher("XXX", Mock())
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [3, 4, 5],
            "volume": [234, 235, 236],
            "average": [1, 2, 3],
            "barCount": [2, 2, 2],
        }
    )
    adjusted = sticher.adjust(df, 1)
    pd.testing.assert_frame_equal(
        adjusted,
        pd.DataFrame(
            {
                "open": [2, 3, 4],
                "high": [3, 4, 5],
                "low": [1, 2, 3],
                "close": [4, 5, 6],
                "volume": [234, 235, 236],
                "average": [2, 3, 4],
                "barCount": [2, 2, 2],
            }
        ),
    )


def test_adjust_negative():
    sticher = FuturesSticher("XXX", Mock())
    df = pd.DataFrame(
        {
            "open": [2, 3, 4],
            "high": [3, 4, 5],
            "low": [1, 2, 3],
            "close": [4, 5, 6],
            "volume": [234, 235, 236],
            "average": [2, 3, 4],
            "barCount": [2, 2, 2],
        }
    )

    adjusted = sticher.adjust(df, -1)
    pd.testing.assert_frame_equal(
        adjusted,
        pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [2, 3, 4],
                "low": [0, 1, 2],
                "close": [3, 4, 5],
                "volume": [234, 235, 236],
                "average": [1, 2, 3],
                "barCount": [2, 2, 2],
            }
        ),
    )


def test_adjust_mul():
    sticher = FuturesSticher("XXX", Mock(), "mul")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [3, 4, 5],
            "volume": [234, 235, 236],
            "average": [1, 2, 3],
            "barCount": [2, 2, 2],
        }
    )
    adjusted = sticher.adjust(df, 2)
    pd.testing.assert_frame_equal(
        adjusted,
        pd.DataFrame(
            {
                "open": [2, 4, 6],
                "high": [4, 6, 8],
                "low": [0, 2, 4],
                "close": [6, 8, 10],
                "volume": [234, 235, 236],
                "average": [2, 4, 6],
                "barCount": [2, 2, 2],
            }
        ),
    )


def test_adjust_mul_less_than_zero():
    sticher = FuturesSticher("XXX", Mock(), "mul")
    df = pd.DataFrame(
        {
            "open": [2.0, 4.0, 6.0],
            "high": [4.0, 6.0, 8.0],
            "low": [0, 2.0, 4.0],
            "close": [6.0, 8.0, 10.0],
            "volume": [234, 235, 236],
            "average": [2.0, 4.0, 6.0],
            "barCount": [2.0, 2.0, 2.0],
        }
    )
    adjusted = sticher.adjust(df, 0.5)
    pd.testing.assert_frame_equal(
        adjusted,
        pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [2.0, 3.0, 4.0],
                "low": [0, 1.0, 2.0],
                "close": [3.0, 4.0, 5.0],
                "volume": [234, 235, 236],
                "average": [1.0, 2.0, 3.0],
                "barCount": [2.0, 2.0, 2.0],
            }
        ),
    )


def test_adjust_none():
    sticher = FuturesSticher("XXX", Mock(), None)
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [3, 4, 5],
            "volume": [234, 235, 236],
            "average": [1, 2, 3],
            "barCount": [2, 2, 2],
        }
    )
    adjusted = sticher.adjust(df, 0)
    pd.testing.assert_frame_equal(adjusted, df)
