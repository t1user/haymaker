# from ib_tools.misc import ContractList

# def test_ContractList_is_a_singleton():
#     ck1 = ContractList()
#     ck2 = ContractList()
#     assert ck1 is ck2


# def test_ContractList_shares_content():
#     ck1 = ContractList()
#     ck2 = ContractList()
#     ck1.append("x")
#     assert "x" in ck2


import pandas as pd
import pytest

from ib_tools.misc import dataframe_signal_extractor


# @pytest.mark.skip(reason="not done yet")
class TestDataframeSignalExtractor:
    """Test dataframe_signal_extractor together with doublewrap."""

    @pytest.fixture
    def df_data(self):
        return pd.DataFrame(
            {
                "price": [123, 123, 125, 124, 128],
                "position": [1, 0, 1, 1, 0],
                "signal": [1, 1, 1, 1, 1],
                "raw": [0, 0, 0, 0, -1],
            }
        )

    def test_no_param_decorator(self, df_data):
        class NoParamTestClass:
            @dataframe_signal_extractor
            def df(self, *args, **kwargs):
                return df_data

        instance = NoParamTestClass()
        assert instance.df() == 1

    def test_with_param_decorator(self, df_data):
        class WithParamTestClass:
            @dataframe_signal_extractor("position")
            def df(self, *args, **kwargs):
                return df_data

        instance = WithParamTestClass()
        assert instance.df() == 0

    def test_with_param_decorator_2(self, df_data):
        class WithParamTestClass2:
            @dataframe_signal_extractor("raw")
            def df(self, *args, **kwargs):
                return df_data

        instance = WithParamTestClass2()
        assert instance.df() == -1
