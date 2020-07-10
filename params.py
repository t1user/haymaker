from trader import Params
from ib_insync import ContFuture


nq = Params(
    contract=ContFuture('NQ', 'GLOBEX'),
    micro_contract=ContFuture('MNQ', 'GLOBEX'),
    trades_per_day=4.5,
    # avg_periods=60,
    volume=12000,
    min_atr=14,
)

es = Params(
    contract=ContFuture('ES', 'GLOBEX'),
    micro_contract=ContFuture('MES', 'GLOBEX'),
    trades_per_day=.8,
    ema_fast=120,
    ema_slow=320,
    sl_atr=3,
    # avg_periods=60,
    volume=43000,
    min_atr=5,
)

gc = Params(
    contract=ContFuture('GC', 'NYMEX'),
    micro_contract=ContFuture('MGC', 'NYMEX'),
    trades_per_day=2,
    ema_fast=60,
    ema_slow=120,
    sl_atr=2,
    # avg_periods=60,
    volume=5500,
    min_atr=1.9,
)

ym = Params(
    contract=ContFuture('YM', 'ECBOT'),
    micro_contract=ContFuture('MYM', 'ECBOT'),
    trades_per_day=1.5,
    ema_fast=60,
    ema_slow=120,
    sl_atr=2,
    # avg_periods=60,
    volume=8000,
    min_atr=55,
)


contracts = [nq, es, ym, gc]


"""
cl = Params(
    contract=('CL', 'NYMEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=2,
    atr_periods=180,
    avg_periods=30,
    # volume=11500,
    alloc=0.05)


rty = Params(
    contract=ContFuture('RTY', 'GLOBEX'),
    micro_contract=ContFuture('M2K', 'GLOBEX'),
    sl_atr=1,
    # avg_periods=30,
    volume=12000,
)


eur = Params(
    contract=('EUR', 'GLOBEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=2,
    atr_periods=180,
    # avg_periods=30,
    volume=6000,
    alloc=0.05)

jpy = Params(
    contract=('JPY', 'GLOBEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=2,
    atr_periods=180,
    # avg_periods=30,
    volume=5500,
    alloc=0.05)
"""


"""
{'EUR': 4020.0,
 'JPY': 2641.0,
 'GC': 5956.0,
 'RTY': 4175.0,
 'YM': 6118.0,
 'NQ': 13356.0,
 'ES': 41337.0}
"""
