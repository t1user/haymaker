from typing import (Callable, Iterable, Dict, Tuple, List, Optional, Any,
                    Sequence, DefaultDict)

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from pyfolio.timeseries import perf_stats  # type: ignore

from vector_backtester import perf, Results
from signal_converters import sig_pos
from stop import stop_loss


class Optimizer:
    """
    Backtest performance of func with parameters sp_1 and sp_2 over
    prices in df.

    Optimizer object will run the backtests on instantiation and make
    results available as properties.

    Args:
    -----

    df - must have columns 'open' and 'close' (signal is generated on
    close, transactions executed on next open)

    func - must take exactly two parameters and return a continuous
    signal (transaction will be executed when signal changes, on index
    point subsequent to the change)

    sp_1 and sp_2 - given as tuples (start, step, mode), where mode is
    either 'geo' for geometric progression or 'lin' for linear
    progression, by default: 'geo'.  For 'geo' mode, if start is an
    int, progression elements will also be cast into int.

    opti_params - explicitly specify which paramaters are to be
    optimized, if not given, two optimization paramaters will be pass
    to func as positional arguments, order in which opti_params are
    given is meaningful; if func is an OptiWrapper, opti_params are
    ignored

    slip - transaction cost expressed as percentage of ticksize

    pairs - pairs of parameters to run the backtest on, if given, sp_1
    and sp_2 will be ignored

    multiprocess - whether simulation is to be run in single or multi
    processes

    Properties:
    -----------

    corr - correlation between returns of various backtests

    rank - 20 best backtests ranked by total returns

    return_mean - mean annual return of all backtests

    return_median - median of annual return of all backtests

    combine_stats - stats for a strategy that's equal weight
    combination of all backtests

    combine_paths - return path of all backtestest equal weighted
    """

    def __init__(self, df: pd.DataFrame, func: Callable,
                 sp_1: Tuple[float, float, str] = (100, 1.25, 'geo'),
                 sp_2: Tuple[float, float, str] = (.1, .1, 'lin'),
                 opti_params: Optional[Sequence[str]] = None,
                 slip=1.5,
                 pairs: Optional[Sequence[Tuple[float, float]]] = None,
                 multiprocess: bool = True
                 ) -> None:

        assert pairs or (sp_1 and sp_2), 'Either pairs or parameters required'

        self.func = func
        self.df = df
        self.slip = slip

        if opti_params and not isinstance(self.func, OptiWrapper):
            assert len(opti_params) == 2, ('Need exactly two optimization'
                                           'parameters')
            self.opti_params = opti_params
        else:
            self.opti_params = []

        self.raw_stats: Dict[Tuple[float, float], pd.Series] = {}
        self.raw_dailys: Dict[Tuple[float, float], pd.DataFrame] = {}
        self.raw_positions: Dict[Tuple[float, float], pd.DataFrame] = {}
        self.raw_dfs: Dict[Tuple[float, float], pd.DataFrame] = {}

        self.pairs = pairs or self.get_pairs(
            self.progression(sp_1), self.progression(sp_2))

        self._table: Dict[str, pd.DataFrame] = {}

        if multiprocess:
            with ProcessPoolExecutor() as executor:
                results = {pair: data for pair, data in
                           zip(self.pairs, executor.map(self.calc, self.pairs))
                           }
            self.bulk_save(results)
        else:
            for p in self.pairs:
                self.save(p, self.calc(p))

        self.extract_stats()
        self.__dict__.update(self._table)
        self.extract_dailys()

    @staticmethod
    def progression(sp: Tuple[Any, ...]) -> Sequence:
        if len(sp) == 3:
            start, step, mode = sp
        elif len(sp) == 2:
            start, step = sp
            mode = 'geo'
        else:
            raise ValueError(
                f'Wrong parameter: {sp}. '
                f'Must be a tuple of: (start, stop, [mode])')

        if isinstance(start, Sequence):
            return start

        if mode == 'geo':
            _t = tuple((start * step**i) for i in range(10))
            if isinstance(start, int):
                return tuple(int(i) for i in _t)
            else:
                return _t
        elif mode == 'lin':
            return tuple(round(start + step*i, 5) for i in range(10))
        else:
            raise ValueError(f"Wrong mode: {mode}, "
                             f"should be 'lin' for linear or 'geo' for "
                             f"geometric")

    @staticmethod
    def get_pairs(sp_1: Iterable[float], sp_2: Iterable[float],
                  ) -> List[Tuple[float, float]]:

        return [(p_1, p_2) for p_1 in sp_1 for p_2 in sp_2]

    def args_kwargs(self, p_1: float, p_2: float) -> Tuple[
            List[float], Dict[str, float]]:
        if self.opti_params:
            kwargs = {key: value for key, value in zip(
                self.opti_params, (p_1, p_2))}
            args = []
        else:
            kwargs = {}
            args = [p_1, p_2]
        return args, kwargs

    def calc(self, pair: Tuple[float, float]) -> Results:
        p_1, p_2 = pair
        args, kwargs = self.args_kwargs(p_1, p_2)
        if isinstance(self.func, OptiWrapper):
            # stop requires position and df, also returns df
            data = self.func(self.df, *args, **kwargs)
            out = perf(data['price'], data['position'], slippage=self.slip)
        else:
            # indicator requires signal and series
            data = sig_pos(self.func(self.df['close'], *args, **kwargs))
            out = perf(self.df['open'], data, slippage=self.slip)
        return out

    def bulk_save(self, data: Dict[Tuple[float, float], Results]) -> None:
        for k, v in data.items():
            self.save(k, v)

    def save(self, pair: Tuple[float, float], out: Results) -> None:
        self.raw_stats[pair] = out.stats
        self.raw_dailys[pair] = out.daily
        self.raw_positions[pair] = out.positions
        self.raw_dfs[pair] = out.df

    def extract_stats(self) -> None:
        self._fields = [i for i in self.raw_stats[self.pairs[-1]].index]

        self.field_trans = {i: i.lower().replace(
            ' ', '_').replace('/', '_').replace('.', '') for i in self._fields}
        self.fields = list(self.field_trans.values())
        dtypes = {self.field_trans[i]: type(
            self.raw_stats[self.pairs[-1]].loc[i]) for i in self._fields}
        self._table = {f: pd.DataFrame() for f in self.fields}
        for index, stats_table in self.raw_stats.items():
            for field in self._fields:
                self._table[self.field_trans[field]
                            ].loc[index] = stats_table[field]

        # cast dfs back to original type (otherwise they're all floats)
        for key, table in self._table.copy().items():
            try:
                self._table[key] = table.fillna(0).astype(dtypes[key])
            except TypeError:
                if dtypes[key] == pd.Timedelta:
                    self._table[key] = table / pd.Timedelta('1day')

    def extract_dailys(self) -> None:
        log_returns = {}
        returns = {}
        paths = {}
        for k, v in self.raw_dailys.items():
            log_returns[k] = v['lreturn']
            returns[k] = v['returns']
            paths[k] = v['balance']
        self.log_returns = pd.DataFrame(log_returns)
        self.returns = pd.DataFrame(returns)
        self.paths = pd.DataFrame(paths)

    @property
    def corr(self) -> pd.DataFrame:
        return self.log_returns.corr()

    @property
    def rank(self) -> pd.Series:
        return (self.paths.iloc[-1] - 1).sort_values().tail(20)

    @property
    def return_mean(self) -> str:
        m = self._table['annual_return'][self._table[
            'annual_return'] != 0].mean().mean()
        return f'{m:.2%}'

    @property
    def return_median(self) -> str:
        m = self._table['annual_return'][self._table[
            'annual_return'] != 0].stack().median()
        return f'{m:.2%}'

    @property
    def combine(self):
        return self.returns.mean(axis=1)

    @property
    def combine_stats(self):
        return perf_stats(self.combine)

    @property
    def combine_paths(self):
        return (self.combine + 1).cumprod()

    def __repr__(self):
        return f'{self.__class__.__name__} for {self.func.__name__}'

    def __str__(self):
        return f"TWo param simulation for {self.func.__name__}"


class OptiWrapper:

    """
    Wrap signal function and stop loss to deliever a callable object
    that can be fed into Optimizer.

    Args:
    -----

    func - signal function, on which stop is to be applied, this
    function must return signal

    X, Y - two optimization (must be exactly two) parameters expressed
    as: 'signal__<param>' or 'stop__<param>'

    Additionally, if optimization is to be run with non-default kwargs
    (for signal function or stop function), those non-default values
    must be given by setting following properties on the object:
    signal_kwargs, stop_kwargs.

    After initialization object can be passed as Optimizer (as func)
    or alternatively, running object's 'optimize' method will return
    Optimizer object.
    """

    signal_kwargs: Dict[Any, Any] = {}
    stop_kwargs: Dict[Any, Any] = {}

    def __init__(self, func: Callable, X: str, Y: str):
        self.X = X
        self.Y = Y
        self.func = func
        self.opti_params_dict = self.extractor((X, Y))
        for i in (X, Y):
            assert '__' in i, ("optimization parameters must be given "
                               "as 'signal__<param>' or 'stop__<param>'")
        self.key_param: List[Tuple[str, str]] = []
        self.params_formater()

    @staticmethod
    def extractor(i: Tuple[str, str]) -> DefaultDict[str, List[str]]:
        """
        Convert user defined optimization parameters in the format
        'signal__<param>' or 'stop__<param>' into a dict that can to
        used to insert the params into appropriate function during
        simulation.
        """
        d: DefaultDict[str, List[str]] = defaultdict(list)
        for x in i:
            items = x.split('__')
            d[items[0]].append(items[1])
        assert set(d.keys()).issubset(set(('signal', 'stop'))), (
            "prefixes must be either 'signal' or 'stop'")
        # create empty lists for missing keys
        d['signal']
        d['stop']
        return d

    def params_formater(self):
        """
        Create a dictionary that will be feed as **params to signal
        and stop functions with placeholders for variable params in
        appropriate places.
        """
        self.params_values_dict: Dict[str, Dict[str, float]] = {
            k: {} for k in self.opti_params_dict.keys()}
        for key, param_list in self.opti_params_dict.items():
            for param in param_list:
                self.params_values_dict[key][param] = 0
                self.key_param.append((key, param))

    def assign(self, X, Y):
        """
        During every param iteration put the current value of params
        into the dict that will feed them into appropriate function.
        """
        for i, j in zip(self.key_param, (X, Y)):
            self.params_values_dict[i[0]][i[1]] = j
        return self.params_values_dict

    def __call__(self, df, X, Y):
        params_values = self.assign(X, Y)
        df['position'] = sig_pos(self.func(df['close'], **self.signal_kwargs,
                                           **params_values['signal']))
        return stop_loss(df, return_type=2, **self.stop_kwargs,
                         **params_values['stop'])

    def optimize(self, df: pd.DataFrame, sp_1, sp_2) -> Optimizer:
        return Optimizer(df, self,
                         sp_1, sp_2,
                         slip=1.5)

    @property
    def __name__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.__class__.__name__} with params: {self.__dict__}'


def plot_grid(data: Optimizer, fields: List[str] = [
        'annual_return', 'sharpe_ratio']) -> None:

    if isinstance(fields, str):
        fields = ['annual_return', fields]

    assert isinstance(fields, Sequence
                      ), f'{fields} is neither string nor sequence'

    assert set(fields).issubset(set(data.fields)), (
        f'Wrong field. '
        f'Allowed fields are: {data.fields}')

    table_one = getattr(data, fields[0])
    table_two = getattr(data, fields[1])

    # percentage of positive rows and columns
    pos_rows = (
        (table_one[table_one > 0].count() / table_one.count()) * 100
    ).astype(int)
    pos_columns = (
        (table_one[table_one > 0].count(axis=1)/table_one.count(axis=1)) * 100
    ).astype(int)

    sns.set_style('whitegrid')
    colormap = sns.diverging_palette(10, 133, n=5, as_cmap=True)
    widths = [1, 1, 1, 10, 10, 1, 1]
    heights = [10, 1, 1, 1]
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(4, 7, width_ratios=widths, height_ratios=heights)

    heatmap_kwargs = {'square': True, 'cmap': colormap, 'annot': True,
                      'annot_kws': {'fontsize': 'large'},
                      'cbar': False, 'linewidth': .3, }
    no_labels = {'xticklabels': False, 'yticklabels': False}

    def formater(field: str, table: pd.DataFrame) -> Dict[str, Any]:
        kwargs_dict: Dict[str, Any] = {}
        if field in ['annual_return', 'sharpe_ratio', 'cumulative_returns',
                     'calmar_ratio', 'sortino_ratio', 'skew',  'position_ev',
                     'monthly_ev', 'annual_ev', 'long_ev', 'short_ev',
                     ]:
            kwargs_dict['center'] = 0

        if table.dtypes.iloc[0] == int:
            kwargs_dict['fmt'] = '.0f'
        else:
            kwargs_dict['fmt'] = '.2f'

        if field in ['annual_return', 'cummulative_return', 'max_drawdown',
                     'daily_value_at_risk', 'win_percent', ]:
            kwargs_dict['fmt'] = '.0%'

        if field == 'annual_return':
            kwargs_dict.update({'vmin': -.3, 'vmax': .3})
        elif field == 'sharpe_ratio':
            kwargs_dict.update({'vmin': -1, 'vmax': 1})
        elif field == 'sortino_ratio':
            kwargs_dict.update({'vmin': -2, 'vmax': 2})
        elif field == 'positions':
            kwargs_dict.update({'center': 250, 'vmin': 0, 'vmax': 750})
        elif field == 'trades':
            kwargs_dict.update({'center': 500, 'vmin': 0, 'vmax': 1500})
        else:
            kwargs_dict['robust'] = True

        return kwargs_dict

    table_1_kwargs = formater(fields[0], table_one)
    table_2_kwargs = formater(fields[1], table_two)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title('%>0')
    sns.heatmap(pd.DataFrame(pos_columns), **heatmap_kwargs,
                **no_labels, fmt=".0f", vmin=0, vmax=100, center=50)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title('mean')
    sns.heatmap(pd.DataFrame(table_one.mean(axis=1)), **
                heatmap_kwargs, **no_labels, **table_1_kwargs)

    ax15 = fig.add_subplot(gs[0, 2])
    ax15.set_title('median')
    sns.heatmap(pd.DataFrame(table_one.median(axis=1)),
                **heatmap_kwargs, **no_labels, **table_1_kwargs)

    ax2 = fig.add_subplot(gs[0, 3])
    ax2.set_title(fields[0])
    sns.heatmap(table_one, **heatmap_kwargs, **table_1_kwargs)

    ax3 = fig.add_subplot(gs[0, 4])
    ax3.set_title(fields[1])
    sns.heatmap(table_two, **heatmap_kwargs, **table_2_kwargs)

    ax35 = fig.add_subplot(gs[0, 5])
    ax35.set_title('median')
    sns.heatmap(pd.DataFrame(table_two.median(axis=1)), **
                heatmap_kwargs, **no_labels, **table_2_kwargs)

    ax4 = fig.add_subplot(gs[0, 6])
    ax4.set_title('mean')
    sns.heatmap(pd.DataFrame(table_two.mean(axis=1)), **
                heatmap_kwargs, **no_labels, **table_2_kwargs)

    ax45 = fig.add_subplot(gs[1, 3])
    ax45.set_title('median')
    sns.heatmap(pd.DataFrame(table_one.median()).T, **
                heatmap_kwargs, **no_labels, **table_1_kwargs)

    ax455 = fig.add_subplot(gs[1, 4])
    sns.heatmap(pd.DataFrame(table_two.median()).T, **
                heatmap_kwargs, **no_labels, **table_2_kwargs)
    ax455.set_title('median')

    ax5 = fig.add_subplot(gs[2, 3])
    ax5.set_title('mean')
    sns.heatmap(pd.DataFrame(table_one.mean()).T, **heatmap_kwargs,
                **no_labels, **table_1_kwargs)

    ax6 = fig.add_subplot(gs[2, 4])
    sns.heatmap(pd.DataFrame(table_two.mean()).T, **heatmap_kwargs,
                **no_labels, **table_2_kwargs)
    ax6.set_title('mean')

    ax7 = fig.add_subplot(gs[3, 3])
    ax7.set_title('%>0')
    sns.heatmap(pd.DataFrame(pos_rows).T, **heatmap_kwargs,
                **no_labels, fmt=".0f", vmin=0, vmax=100, center=50)
