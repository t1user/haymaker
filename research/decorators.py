from functools import wraps

import pandas as pd


def verify_input(func):
    """Allow signal producting functions to work with either dataframe or series."""

    @wraps(func)
    def verify(data, *args, **kwargs) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            data = pd.DataFrame({"close": data})
        elif isinstance(data, pd.DataFrame):
            data = data.copy()
        else:
            raise TypeError(
                f"Data must be either Series or DataFrame with column 'close'"
                f" containing prices, not {type(data)}."
            )
        return func(data, *args, **kwargs)

    return verify
