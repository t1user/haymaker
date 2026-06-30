"""Grid-search helpers for dataframe backtesting research.

This module runs a user strategy function over several dataframe/parameter
combinations, sends each returned transaction dataframe through
``haymaker.research.backtester.perf``, and returns a ``GridSearchResult``.

The strategy function receives ``df["close"]`` as its first argument by default.
Set ``pass_full_df=True`` when the strategy needs the full dataframe. Searched
parameters are passed positionally unless ``param_names`` is supplied; use
``fixed_kwargs`` for settings that are shared by every simulation.

Examples:
    Grid search from two progressions:

    .. code-block:: python

        from haymaker.research.grid_search import GridSearch, show_grid_table

        search = GridSearch.from_progressions(
            df,
            strategy_func,
            [(0.5, 0.5, "lin"), (30, 10, "lin")],
            param_names=("threshold", "lookback"),
            fixed_kwargs={"side": "long"},
            multiprocess=False,
        )

        result = search.run()
        show_grid_table(result)

    Grid search from explicit parameter pairs:

    .. code-block:: python

        search = GridSearch.from_pairs(
            df,
            strategy_func,
            [(0.5, 30), (1.0, 40), (1.5, 50)],
            param_names=("threshold", "lookback"),
        )

        result = search.run()

    Grid search over many dataframes with one parameter tuple:

    .. code-block:: python

        search = GridSearch.from_dfs(
            {"sample_a": df_a, "sample_b": df_b},
            strategy_func,
            params=(0.5, 30),
            param_names=("threshold", "lookback"),
        )

        result = search.run()
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from numbers import Real
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib.figure import Figure

ParamPair = tuple[Any, Any]
JobKey = tuple[Hashable, Hashable]
MissingPolicy = Literal["zero", "raise", "drop"]

DEFAULT_GRID_TABLE_FIELDS: list[str] = [
    "annual_return",
    "sharpe_ratio",
    "position_ev",
    "monthly_ev",
    "max_drawdown",
    "skew",
    "win_ratio",
    "payoff_ratio",
    "long_ev",
    "short_ev",
    "calmar_ratio",
    "annual_volatility",
]

__all__ = [
    "GridSearch",
    "GridSearchResult",
    "combined_returns",
    "combined_path",
    "combined_stats",
    "plot_grid",
    "show_grid",
    "show_grid_table",
]


def _format_grid_label(value: Any) -> str:
    """Format grid-search labels without noisy floating-point tails.

    Args:
        value: Row or column label from a grid-search statistics table.

    Returns:
        Compact text representation for notebook table display.
    """
    if isinstance(value, tuple):
        return "(" + ", ".join(_format_grid_label(item) for item in value) + ")"
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    return str(value)


@dataclass(frozen=True)
class Simulation:
    """Single grid-search simulation job.

    Args:
        key: Two-part key used to store and display the simulation result.
        input_data: Data passed as the first argument to the user function.
        args: Searched parameters passed positionally to the user function.
        kwargs: Fixed and searched keyword arguments passed to the user
            function.
    """

    key: JobKey
    input_data: pd.Series | pd.DataFrame
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]


@dataclass
class GridSearchResult:
    """Results produced by :class:`GridSearch`.

    Args:
        raw_stats: Performance statistics keyed by simulation key.
        raw_dailys: Daily return data keyed by simulation key.
        raw_positions: Trade/position records keyed by simulation key.
        raw_dfs: Bar-level performance data keyed by simulation key.
        raw_warnings: Warning messages keyed by simulation key.

    Attributes:
        tables: Statistic tables keyed by translated field name. This is the
            canonical access path for grid-search statistics.
        fields: Translated statistic field names available in ``tables``.
    """

    raw_stats: dict[JobKey, pd.Series]
    raw_dailys: dict[JobKey, pd.DataFrame]
    raw_positions: dict[JobKey, pd.DataFrame]
    raw_dfs: dict[JobKey, pd.DataFrame]
    raw_warnings: dict[JobKey, list[str]]
    tables: dict[str, pd.DataFrame] = field(init=False)
    fields: list[str] = field(init=False)
    field_trans: dict[str, str] = field(init=False)
    dtypes: dict[str, type[Any]] = field(init=False)
    _field_names: list[str] = field(init=False, repr=False)
    _returns: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _log_returns: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _paths: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.raw_stats:
            raise ValueError("GridSearchResult requires at least one result.")
        self._build_stat_tables()

    def __getattr__(self, name: str) -> pd.DataFrame:
        tables = self.__dict__.get("tables", {})
        if name in tables:
            return tables[name]
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def _build_stat_tables(self) -> None:
        self._field_names = list(next(iter(self.raw_stats.values())).index)
        self.field_trans = {
            field: self._translate_field_name(field) for field in self._field_names
        }
        if len(set(self.field_trans.values())) != len(self.field_trans):
            raise ValueError("Statistic field names are not unique after translation.")

        self.fields = list(self.field_trans.values())
        self.tables = {}
        self.dtypes = {}

        for field_name in self._field_names:
            field = self.field_trans[field_name]
            values = {
                key: stats.get(field_name, pd.NA)
                for key, stats in self.raw_stats.items()
            }
            table = pd.Series(values).unstack()
            sample = self._first_valid_value(values.values())
            self.dtypes[field] = type(sample) if sample is not None else object
            self.tables[field] = self._coerce_stat_table(table, sample)

    @staticmethod
    def _translate_field_name(field: str) -> str:
        return field.lower().replace(" ", "_").replace("/", "_").replace(".", "")

    @staticmethod
    def _first_valid_value(values: Iterable[Any]) -> Any | None:
        for value in values:
            if not pd.isna(value):
                return value
        return None

    @staticmethod
    def _coerce_stat_table(table: pd.DataFrame, sample: Any | None) -> pd.DataFrame:
        if sample is None:
            return table
        if isinstance(sample, pd.Timedelta):
            return table.apply(lambda column: column / pd.Timedelta("1day"))
        try:
            return table.astype(type(sample))
        except (TypeError, ValueError):
            return table

    def _daily_table(self, column: str) -> pd.DataFrame:
        if not self.raw_dailys:
            raise ValueError(
                "Daily data is unavailable. Re-run GridSearch with save_mem=False "
                f"to access {column!r}."
            )
        missing = [key for key, daily in self.raw_dailys.items() if column not in daily]
        if missing:
            raise ValueError(
                f"Daily result column {column!r} is missing for {missing}."
            )
        return pd.DataFrame(
            {key: daily[column] for key, daily in self.raw_dailys.items()}
        )

    @property
    def returns(self) -> pd.DataFrame:
        """Daily returns for each simulation."""
        if self._returns is None:
            self._returns = self._daily_table("returns")
        return self._returns

    @property
    def log_returns(self) -> pd.DataFrame:
        """Daily log returns for each simulation."""
        if self._log_returns is None:
            self._log_returns = self._daily_table("lreturn")
        return self._log_returns

    @property
    def paths(self) -> pd.DataFrame:
        """Daily balance paths for each simulation."""
        if self._paths is None:
            self._paths = self._daily_table("balance")
        return self._paths

    @property
    def corr(self) -> pd.DataFrame:
        """Correlation matrix of simulation log returns."""
        return self.log_returns.corr()

    @property
    def rank(self) -> pd.Series:
        """Top 20 simulations by final balance minus one."""
        return (self.paths.iloc[-1] - 1).dropna().sort_values().tail(20)

    @property
    def return_mean(self) -> str:
        """Mean non-zero annual return formatted as a percentage."""
        m = (
            self.tables["annual_return"][self.tables["annual_return"] != 0]
            .mean()
            .mean()
        )
        return f"{m:.2%}"

    @property
    def return_median(self) -> str:
        """Median non-zero annual return formatted as a percentage."""
        m = (
            self.tables["annual_return"][self.tables["annual_return"] != 0]
            .stack()
            .median()
        )
        return f"{m:.2%}"

    @property
    def warnings(self) -> dict[JobKey, list[str]]:
        """Non-empty warnings keyed by simulation key."""
        return {key: value for key, value in self.raw_warnings.items() if value}

    def combined_returns(
        self,
        keys: Sequence[JobKey],
        *,
        missing: MissingPolicy = "zero",
    ) -> pd.Series:
        """Return equal-weight daily returns for selected simulations.

        The combined return stream represents a daily-rebalanced equal-weight
        portfolio of completed simulation return streams. It is not a directly
        simulated strategy path.

        Args:
            keys: Simulation keys to combine.
            missing: Missing-data policy. ``"zero"`` treats missing sleeve
                returns as idle cash, ``"raise"`` rejects missing selected
                returns, and ``"drop"`` preserves the old skip-NaN averaging
                behavior.

        Returns:
            Daily return series for the selected equal-weight sleeve portfolio.
        """
        selected = self._selected_returns(keys)
        if missing == "zero":
            return selected.fillna(0).sum(axis=1) / len(selected.columns)
        if missing == "raise":
            if selected.isna().any().any():
                raise ValueError("Selected simulations contain missing daily returns.")
            return selected.mean(axis=1)
        if missing == "drop":
            return selected.mean(axis=1, skipna=True)
        raise ValueError(f"Unknown missing policy: {missing!r}")

    def combined_path(
        self,
        keys: Sequence[JobKey],
        *,
        missing: MissingPolicy = "zero",
    ) -> pd.Series:
        """Return the cumulative path of selected combined returns.

        Args:
            keys: Simulation keys to combine.
            missing: Missing-data policy forwarded to
                :meth:`combined_returns`.

        Returns:
            Cumulative return path built from the selected combined returns.
        """
        return (self.combined_returns(keys, missing=missing) + 1).cumprod()

    def combined_stats(
        self,
        keys: Sequence[JobKey],
        *,
        missing: MissingPolicy = "zero",
    ) -> pd.Series:
        """Return pyfolio return statistics for selected combined returns.

        ``pyfolio.timeseries.perf_stats`` is imported lazily, so importing this
        module does not load pyfolio unless this method is called.

        Args:
            keys: Simulation keys to combine.
            missing: Missing-data policy forwarded to
                :meth:`combined_returns`.

        Returns:
            Pyfolio statistics for the selected combined return stream.
        """
        from pyfolio.timeseries import perf_stats  # type: ignore[import-untyped]

        return perf_stats(self.combined_returns(keys, missing=missing))

    def _selected_returns(self, keys: Sequence[JobKey]) -> pd.DataFrame:
        if not keys:
            raise ValueError("At least one simulation key is required.")
        missing_keys = [key for key in keys if key not in self.returns.columns]
        if missing_keys:
            raise KeyError(
                f"Simulation keys not found in result returns: {missing_keys}"
            )
        return self.returns.loc[:, list(keys)]


@dataclass
class GridSearch:
    """Build and run a grid search over backtest simulations.

    Use the class methods to construct a search from parameter progressions,
    explicit parameter pairs, or a collection of dataframes. ``run()`` executes
    the user function for each simulation, passes its transaction dataframe to
    :func:`haymaker.research.backtester.perf`, and returns a
    :class:`GridSearchResult`.

    Args:
        func: User function that returns a transaction dataframe accepted by
            ``perf``.
        simulations: Simulation jobs to execute.
        multiprocess: Whether to run simulations in separate processes.
        save_mem: If ``True``, omit daily returns and bar-level data from the
            result.
        perf_kwargs: Keyword arguments passed to ``perf``.
    """

    func: Callable[..., pd.DataFrame]
    simulations: Sequence[Simulation]
    multiprocess: bool = True
    save_mem: bool = False
    perf_kwargs: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.simulations:
            raise ValueError("GridSearch requires at least one simulation.")

    @classmethod
    def from_progressions(
        cls,
        df: pd.DataFrame,
        func: Callable[..., pd.DataFrame],
        progressions: Sequence[tuple[Any, ...]],
        *,
        param_names: Sequence[str] | None = None,
        fixed_kwargs: Mapping[str, Any] | None = None,
        pass_full_df: bool = False,
        multiprocess: bool = True,
        save_mem: bool = False,
        **perf_kwargs: Any,
    ) -> GridSearch:
        """Create a grid search from two parameter progressions.

        Args:
            df: Source dataframe used by every simulation.
            func: Function returning a transaction dataframe accepted by
                ``perf``.
            progressions: Exactly two progression specs. A spec can generate
                ten values as ``(start, step[, mode])``, where ``mode`` is
                ``"geo"`` or ``"lin"``. A spec can also provide explicit values
                by passing a sequence as ``start``; in that case, ``step`` and
                ``mode`` are ignored.
            param_names: Optional names for the two searched parameters. When
                omitted, searched values are passed positionally.
            fixed_kwargs: Keyword arguments included in every call to ``func``.
            pass_full_df: Pass the full dataframe to ``func`` instead of
                ``df["close"]``.
            multiprocess: Whether to run simulations in separate processes.
            save_mem: If ``True``, omit daily returns and bar-level data.
            **perf_kwargs: Keyword arguments passed to ``perf``.
        """
        if len(progressions) != 2:
            raise ValueError("from_progressions requires exactly two progressions.")
        first, second = (cls.progression(spec) for spec in progressions)
        return cls.from_pairs(
            df,
            func,
            cls.get_pairs(first, second),
            param_names=param_names,
            fixed_kwargs=fixed_kwargs,
            pass_full_df=pass_full_df,
            multiprocess=multiprocess,
            save_mem=save_mem,
            **perf_kwargs,
        )

    @classmethod
    def from_pairs(
        cls,
        df: pd.DataFrame,
        func: Callable[..., pd.DataFrame],
        pairs: Sequence[ParamPair],
        *,
        param_names: Sequence[str] | None = None,
        fixed_kwargs: Mapping[str, Any] | None = None,
        pass_full_df: bool = False,
        multiprocess: bool = True,
        save_mem: bool = False,
        **perf_kwargs: Any,
    ) -> GridSearch:
        """Create a grid search from explicit parameter pairs.

        Args:
            df: Source dataframe used by every simulation.
            func: Function returning a transaction dataframe accepted by
                ``perf``.
            pairs: Explicit two-value parameter pairs.
            param_names: Optional names for the two searched parameters. When
                omitted, searched values are passed positionally.
            fixed_kwargs: Keyword arguments included in every call to ``func``.
            pass_full_df: Pass the full dataframe to ``func`` instead of
                ``df["close"]``.
            multiprocess: Whether to run simulations in separate processes.
            save_mem: If ``True``, omit daily returns and bar-level data.
            **perf_kwargs: Keyword arguments passed to ``perf``.
        """
        pair_list = [cls._as_pair(pair) for pair in pairs]
        if not pair_list:
            raise ValueError("from_pairs requires at least one pair.")

        input_data = cls._input_data(df, pass_full_df)
        simulations = []
        for pair in pair_list:
            args, kwargs = cls._call_args(pair, param_names, fixed_kwargs)
            simulations.append(
                Simulation(
                    key=pair,
                    input_data=input_data,
                    args=args,
                    kwargs=kwargs,
                )
            )
        return cls(func, simulations, multiprocess, save_mem, dict(perf_kwargs))

    @classmethod
    def from_dfs(
        cls,
        dfs: Mapping[Hashable, pd.DataFrame] | Sequence[pd.DataFrame],
        func: Callable[..., pd.DataFrame],
        params: Sequence[Any],
        *,
        param_names: Sequence[str] | None = None,
        fixed_kwargs: Mapping[str, Any] | None = None,
        pass_full_df: bool = False,
        multiprocess: bool = True,
        save_mem: bool = False,
        **perf_kwargs: Any,
    ) -> GridSearch:
        """Create a grid search over many dataframes and one parameter tuple.

        Args:
            dfs: Mapping or sequence of source dataframes. Mapping keys become
                dataframe labels; sequence inputs use integer labels.
            func: Function returning a transaction dataframe accepted by
                ``perf``.
            params: Fixed searched parameters used for every dataframe.
            param_names: Optional names for ``params``. When omitted, values are
                passed positionally.
            fixed_kwargs: Keyword arguments included in every call to ``func``.
            pass_full_df: Pass each full dataframe to ``func`` instead of its
                ``"close"`` column.
            multiprocess: Whether to run simulations in separate processes.
            save_mem: If ``True``, omit daily returns and bar-level data.
            **perf_kwargs: Keyword arguments passed to ``perf``.
        """
        params_tuple = tuple(params)
        args, kwargs = cls._call_args(params_tuple, param_names, fixed_kwargs)
        simulations = []
        for label, df in cls._iter_dataframes(dfs):
            key = (label, params_tuple)
            cls._validate_hashable_key(key)
            simulations.append(
                Simulation(
                    key=key,
                    input_data=cls._input_data(df, pass_full_df),
                    args=args,
                    kwargs=dict(kwargs),
                )
            )
        if not simulations:
            raise ValueError("from_dfs requires at least one dataframe.")
        return cls(func, simulations, multiprocess, save_mem, dict(perf_kwargs))

    @staticmethod
    def progression(spec: tuple[Any, ...]) -> tuple[Any, ...]:
        """Create ten values from a linear or geometric progression spec."""
        if len(spec) == 3:
            start, step, mode = spec
        elif len(spec) == 2:
            start, step = spec
            mode = "geo"
        else:
            raise ValueError(
                f"Wrong progression: {spec}. Must be (start, step[, mode])."
            )

        if isinstance(start, Sequence) and not isinstance(start, (str, bytes)):
            return tuple(start)

        if mode == "geo":
            values = tuple(start * step**i for i in range(10))
            return (
                tuple(int(value) for value in values)
                if isinstance(start, int)
                else values
            )
        if mode == "lin":
            return tuple(round(start + step * i, 5) for i in range(10))
        raise ValueError(f"Wrong mode: {mode}. Use 'lin' or 'geo'.")

    @staticmethod
    def get_pairs(sp_1: Iterable[Any], sp_2: Iterable[Any]) -> list[ParamPair]:
        """Return the Cartesian product of two parameter sequences."""
        return [(p_1, p_2) for p_1 in sp_1 for p_2 in sp_2]

    @staticmethod
    def _as_pair(pair: Sequence[Any]) -> ParamPair:
        if isinstance(pair, (str, bytes)):
            raise ValueError(
                "Parameter pairs must be two-value sequences, not strings."
            )
        values = tuple(pair)
        if len(values) != 2:
            raise ValueError(f"Parameter pair must contain exactly two values: {pair}")
        GridSearch._validate_hashable_key(values)
        return values

    @staticmethod
    def _input_data(
        df: pd.DataFrame,
        pass_full_df: bool,
    ) -> pd.Series | pd.DataFrame:
        if pass_full_df:
            return df
        if "close" not in df:
            raise ValueError("Dataframe must contain 'close' when pass_full_df=False.")
        return df["close"]

    @staticmethod
    def _validated_fixed_kwargs(
        fixed_kwargs: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if fixed_kwargs is None:
            return {}
        invalid = [key for key in fixed_kwargs if not isinstance(key, str)]
        if invalid:
            raise ValueError(f"fixed_kwargs keys must be strings: {invalid}")
        return dict(fixed_kwargs)

    @staticmethod
    def _validated_param_names(
        param_names: Sequence[str] | None,
        expected_count: int,
    ) -> tuple[str, ...] | None:
        if param_names is None:
            return None
        names = tuple(param_names)
        if len(names) != expected_count:
            raise ValueError(
                f"Expected {expected_count} parameter names, got {len(names)}."
            )
        invalid = [name for name in names if not isinstance(name, str)]
        if invalid:
            raise ValueError(f"Parameter names must be strings: {invalid}")
        return names

    @staticmethod
    def _call_args(
        params: Sequence[Any],
        param_names: Sequence[str] | None,
        fixed_kwargs: Mapping[str, Any] | None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        values = tuple(params)
        fixed = GridSearch._validated_fixed_kwargs(fixed_kwargs)
        names = GridSearch._validated_param_names(param_names, len(values))
        if names is None:
            return values, fixed

        overlap = set(names) & set(fixed)
        if overlap:
            raise ValueError(
                "Searched parameter names overlap with fixed_kwargs: "
                f"{sorted(overlap)}"
            )
        return (), {**fixed, **dict(zip(names, values))}

    @staticmethod
    def _iter_dataframes(
        dfs: Mapping[Hashable, pd.DataFrame] | Sequence[pd.DataFrame],
    ) -> Iterable[tuple[Hashable, pd.DataFrame]]:
        if isinstance(dfs, pd.DataFrame):
            raise ValueError("from_dfs expects a mapping or sequence of dataframes.")

        if isinstance(dfs, Mapping):
            for label, df in dfs.items():
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(
                        f"Expected dataframe for {label!r}, got {type(df)!r}."
                    )
                GridSearch._validate_hashable_key(label)
                yield label, df
            return

        for label, df in enumerate(dfs):
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected dataframe for {label!r}, got {type(df)!r}.")
            yield label, df

    @staticmethod
    def _validate_hashable_key(value: Any) -> None:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(
                f"Simulation key values must be hashable: {value!r}"
            ) from exc

    @staticmethod
    def _result_from_items(
        items: Sequence[tuple[JobKey, Any]],
        *,
        save_mem: bool,
    ) -> GridSearchResult:
        raw_stats: dict[JobKey, pd.Series] = {}
        raw_dailys: dict[JobKey, pd.DataFrame] = {}
        raw_positions: dict[JobKey, pd.DataFrame] = {}
        raw_dfs: dict[JobKey, pd.DataFrame] = {}
        raw_warnings: dict[JobKey, list[str]] = {}

        for key, out in items:
            raw_stats[key] = out.stats
            raw_positions[key] = out.positions
            raw_warnings[key] = out.warnings
            if not save_mem:
                raw_dailys[key] = out.daily
                raw_dfs[key] = out.df

        return GridSearchResult(
            raw_stats=raw_stats,
            raw_dailys=raw_dailys,
            raw_positions=raw_positions,
            raw_dfs=raw_dfs,
            raw_warnings=raw_warnings,
        )

    @staticmethod
    def _callable_name(func: Callable[..., Any]) -> str:
        if isinstance(func, partial):
            return GridSearch._callable_name(func.func)
        return getattr(func, "__name__", repr(func))

    def run(self) -> GridSearchResult:
        """Execute all simulations and return a result object."""
        runner = partial(
            _run_simulation,
            self.func,
            dict(self.perf_kwargs),
            self.save_mem,
        )
        if self.multiprocess:
            with ProcessPoolExecutor() as executor:
                items = list(executor.map(runner, self.simulations))
        else:
            items = [runner(simulation) for simulation in self.simulations]
        return self._result_from_items(items, save_mem=self.save_mem)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(func={self._callable_name(self.func)}, "
            f"simulations={len(self.simulations)})"
        )


def _run_simulation(
    func: Callable[..., pd.DataFrame],
    perf_kwargs: dict[str, Any],
    save_mem: bool,
    simulation: Simulation,
) -> tuple[JobKey, Any]:
    from .backtester import Results, perf

    try:
        data = func(simulation.input_data, *simulation.args, **simulation.kwargs)
    except Exception:
        print(f"Error running function for key: {simulation.key}")
        raise

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "GridSearch functions must return a transaction dataframe accepted "
            "by perf()."
        )

    try:
        out = perf(data, **perf_kwargs)
    except Exception:
        print(f"Error running perf for key: {simulation.key}")
        raise

    if save_mem:
        out = Results(
            out.stats,
            pd.DataFrame(),
            out.positions,
            pd.DataFrame(),
            out.warnings,
        )
    return simulation.key, out


def combined_returns(
    result: GridSearchResult,
    keys: Sequence[JobKey],
    *,
    missing: MissingPolicy = "zero",
) -> pd.Series:
    """Return equal-weight daily returns for selected simulations.

    The combined return stream represents a daily-rebalanced equal-weight
    portfolio of completed simulation return streams. It is not a directly
    simulated strategy path.

    Args:
        result: Grid-search result containing daily simulation returns.
        keys: Simulation keys to combine.
        missing: Missing-data policy. ``"zero"`` treats missing sleeve returns
            as idle cash, ``"raise"`` rejects missing selected returns, and
            ``"drop"`` preserves the old skip-NaN averaging behavior.

    Returns:
        Daily return series for the selected equal-weight sleeve portfolio.
    """
    return result.combined_returns(keys, missing=missing)


def combined_path(
    result: GridSearchResult,
    keys: Sequence[JobKey],
    *,
    missing: MissingPolicy = "zero",
) -> pd.Series:
    """Return the cumulative path of selected equal-weight simulation returns.

    Args:
        result: Grid-search result containing daily simulation returns.
        keys: Simulation keys to combine.
        missing: Missing-data policy forwarded to :func:`combined_returns`.

    Returns:
        Cumulative return path built from the selected combined returns.
    """
    return result.combined_path(keys, missing=missing)


def combined_stats(
    result: GridSearchResult,
    keys: Sequence[JobKey],
    *,
    missing: MissingPolicy = "zero",
) -> pd.Series:
    """Return pyfolio return statistics for selected combined returns.

    ``pyfolio.timeseries.perf_stats`` is imported lazily, so importing this
    module does not load pyfolio unless this function is called.

    Args:
        result: Grid-search result containing daily simulation returns.
        keys: Simulation keys to combine.
        missing: Missing-data policy forwarded to :func:`combined_returns`.

    Returns:
        Pyfolio statistics for the selected combined return stream.
    """
    return result.combined_stats(keys, missing=missing)


def plot_grid(
    data: GridSearchResult,
    fields: str | Sequence[str] = ("annual_return", "sharpe_ratio"),
) -> Figure:
    """Create the notebook heatmap layout for two grid-search result fields.

    Args:
        data: Grid-search result object containing statistic tables.
        fields: Statistic field or pair of fields to plot. A string keeps the
            notebook convention of plotting ``annual_return`` next to that
            field.

    Returns:
        Matplotlib figure containing the grid heatmaps and row/column summary
        heatmaps.

    Raises:
        AssertionError: If a requested field is unavailable, or if either
            selected field is not a 10 by 10 table.
    """
    if isinstance(fields, str):
        fields = ["annual_return", fields]
    else:
        fields = list(fields)

    assert isinstance(fields, Sequence), f"{fields} is neither string nor sequence"

    assert set(fields).issubset(set(data.fields)), (
        f"Wrong field. " f"Allowed fields are: {data.fields}"
    )

    table_one = getattr(data, fields[0])
    table_two = getattr(data, fields[1])

    for t in [table_one, table_two]:
        assert t.shape == (10, 10), f"Wrong data shape {t.shape}, must be (10, 10)"

    # percentage of positive rows and columns
    pos_rows = ((table_one[table_one > 0].count() / table_one.count()) * 100).astype(
        int
    )
    pos_columns = (
        (table_one[table_one > 0].count(axis=1) / table_one.count(axis=1)) * 100
    ).astype(int)

    sns.set_style("whitegrid")
    colormap = sns.diverging_palette(10, 133, n=5, as_cmap=True)
    widths = [1, 1, 1, 10, 10, 1, 1]
    heights = [10, 1, 1, 1]
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(4, 7, width_ratios=widths, height_ratios=heights)

    heatmap_kwargs: dict[str, Any] = {
        "square": True,
        "cmap": colormap,
        "annot": True,
        "annot_kws": {"fontsize": "large"},
        "cbar": False,
        "linewidth": 0.3,
    }
    no_labels: dict[str, Any] = {"xticklabels": False, "yticklabels": False}

    def formater(field: str, table: pd.DataFrame) -> dict[str, Any]:
        kwargs_dict: dict[str, Any] = {}
        if field in [
            "annual_return",
            "sharpe_ratio",
            "cumulative_returns",
            "calmar_ratio",
            "sortino_ratio",
            "skew",
            "position_ev",
            "monthly_ev",
            "annual_ev",
            "long_ev",
            "short_ev",
        ]:
            kwargs_dict["center"] = 0

        if table.dtypes.iloc[0] == int:
            kwargs_dict["fmt"] = ".0f"
        else:
            kwargs_dict["fmt"] = ".2f"

        if field in [
            "annual_return",
            "cummulative_return",
            "max_drawdown",
            "daily_value_at_risk",
            "win_percent",
        ]:
            kwargs_dict["fmt"] = ".0%"

        if field == "annual_return":
            kwargs_dict.update({"vmin": -0.3, "vmax": 0.3})
        elif field == "sharpe_ratio":
            kwargs_dict.update({"vmin": -1, "vmax": 1})
        elif field == "sortino_ratio":
            kwargs_dict.update({"vmin": -2, "vmax": 2})
        elif field == "positions":
            kwargs_dict.update({"center": 250, "vmin": 0, "vmax": 750})
        elif field == "trades":
            kwargs_dict.update({"center": 500, "vmin": 0, "vmax": 1500})
        else:
            kwargs_dict["robust"] = True

        return kwargs_dict

    table_1_kwargs = formater(fields[0], table_one)
    table_2_kwargs = formater(fields[1], table_two)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("%>0")
    sns.heatmap(
        pd.DataFrame(pos_columns),
        **heatmap_kwargs,
        **no_labels,
        fmt=".0f",
        vmin=0,
        vmax=100,
        center=50,
    )

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("mean")
    sns.heatmap(
        pd.DataFrame(table_one.mean(axis=1)),
        **heatmap_kwargs,
        **no_labels,
        **table_1_kwargs,
    )

    ax15 = fig.add_subplot(gs[0, 2])
    ax15.set_title("median")
    sns.heatmap(
        pd.DataFrame(table_one.median(axis=1)),
        **heatmap_kwargs,
        **no_labels,
        **table_1_kwargs,
    )

    ax2 = fig.add_subplot(gs[0, 3])
    ax2.set_title(fields[0])
    sns.heatmap(table_one, **heatmap_kwargs, **table_1_kwargs)

    ax3 = fig.add_subplot(gs[0, 4])
    ax3.set_title(fields[1])
    sns.heatmap(table_two, **heatmap_kwargs, **table_2_kwargs)

    ax35 = fig.add_subplot(gs[0, 5])
    ax35.set_title("median")
    sns.heatmap(
        pd.DataFrame(table_two.median(axis=1)),
        **heatmap_kwargs,
        **no_labels,
        **table_2_kwargs,
    )

    ax4 = fig.add_subplot(gs[0, 6])
    ax4.set_title("mean")
    sns.heatmap(
        pd.DataFrame(table_two.mean(axis=1)),
        **heatmap_kwargs,
        **no_labels,
        **table_2_kwargs,
    )

    ax45 = fig.add_subplot(gs[1, 3])
    ax45.set_title("median")
    sns.heatmap(
        pd.DataFrame(table_one.median()).T,
        **heatmap_kwargs,
        **no_labels,
        **table_1_kwargs,
    )

    ax455 = fig.add_subplot(gs[1, 4])
    sns.heatmap(
        pd.DataFrame(table_two.median()).T,
        **heatmap_kwargs,
        **no_labels,
        **table_2_kwargs,
    )
    ax455.set_title("median")

    ax5 = fig.add_subplot(gs[2, 3])
    ax5.set_title("mean")
    sns.heatmap(
        pd.DataFrame(table_one.mean()).T,
        **heatmap_kwargs,
        **no_labels,
        **table_1_kwargs,
    )

    ax6 = fig.add_subplot(gs[2, 4])
    sns.heatmap(
        pd.DataFrame(table_two.mean()).T,
        **heatmap_kwargs,
        **no_labels,
        **table_2_kwargs,
    )
    ax6.set_title("mean")

    ax7 = fig.add_subplot(gs[3, 3])
    ax7.set_title("%>0")
    sns.heatmap(
        pd.DataFrame(pos_rows).T,
        **heatmap_kwargs,
        **no_labels,
        fmt=".0f",
        vmin=0,
        vmax=100,
        center=50,
    )

    return fig


def show_grid(
    data: GridSearchResult,
    fields: str | Sequence[str] = ("annual_return", "sharpe_ratio"),
    plotting_function: Callable[
        [GridSearchResult, str | Sequence[str]],
        Figure | None,
    ] = plot_grid,
) -> None:
    """Display a grid-search plot in a notebook and close the figure.

    Args:
        data: Grid-search result object passed to ``plotting_function``.
        fields: Statistic field or pair of fields to plot.
        plotting_function: Function that creates the grid figure. It should
            accept the same arguments as ``plot_grid`` and preferably return the
            created figure. If it returns ``None``, the current matplotlib figure
            is displayed and closed.
    """
    fig = plotting_function(data, fields)
    if fig is None:
        fig = plt.gcf()

    from IPython.display import display

    display(fig)
    plt.close(fig)


def show_grid_table(
    data: GridSearchResult,
    fields: str | Sequence[str] | None = None,
) -> None:
    """Display grid-search result fields as plain formatted notebook tables.

    Args:
        data: Grid-search result object containing statistic tables.
        fields: Statistic field or fields to display. ``None`` displays the
            preferred default field set from ``DEFAULT_GRID_TABLE_FIELDS``. A
            string keeps the same convention as ``plot_grid`` and displays
            ``annual_return`` plus the requested field.
    """
    if fields is None:
        fields = DEFAULT_GRID_TABLE_FIELDS.copy()
    elif isinstance(fields, str):
        fields = ["annual_return", fields]
    else:
        fields = list(fields)

    assert isinstance(fields, Sequence), f"{fields} is neither string nor sequence"
    assert set(fields).issubset(
        set(data.fields)
    ), f"Wrong field. Allowed fields are: {data.fields}"

    percent_fields = {
        "annual_return",
        "cummulative_return",
        "max_drawdown",
        "daily_value_at_risk",
        "win_percent",
    }

    from IPython.display import display

    for field in fields:
        table = getattr(data, field)
        if table.dtypes.iloc[0] == int:
            fmt = "{:.0f}"
        else:
            fmt = "{:.2f}"

        if field in percent_fields:
            fmt = "{:.0%}"

        styled_table = (
            table.style.format(fmt)
            .format_index(_format_grid_label, axis=0)
            .format_index(_format_grid_label, axis=1)
            .set_caption(field)
        )
        display(styled_table)
