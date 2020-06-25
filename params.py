from trader import Params

nq = Params(
    contract=('NQ', 'GLOBEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=1,
    atr_periods=180,
    # avg_periods=60,
    volume=13000,
    alloc=0.3)

es = Params(
    contract=('ES', 'GLOBEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=120,
    ema_slow=320,
    sl_atr=3,
    atr_periods=180,
    # avg_periods=60,
    volume=43000,
    alloc=0.3)

gc = Params(
    contract=('GC', 'NYMEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=3,
    atr_periods=180,
    # avg_periods=30,
    volume=6000,
    alloc=0.35)

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

eur = Params(
    contract=('EUR', 'GLOBEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=2,
    atr_periods=180,
    # avg_periods=30,
    volume=4000,
    alloc=0.05)

jpy = Params(
    contract=('JPY', 'GLOBEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=2,
    atr_periods=180,
    # avg_periods=30,
    volume=2600,
    alloc=0.05)

rty = Params(
    contract=('RTY', 'GLOBEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=2,
    atr_periods=180,
    # avg_periods=30,
    volume=4200,
    alloc=0.05)

ym = Params(
    contract=('YM', 'ECBOT'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=2,
    atr_periods=180,
    # avg_periods=30,
    volume=6100,
    alloc=0.05)


contracts = [nq, es, eur, jpy, gc, rty, ym]


"""
{'EUR': 4020.0,
 'JPY': 2641.0,
 'GC': 5956.0,
 'RTY': 4175.0,
 'YM': 6118.0,
 'NQ': 13356.0,
 'ES': 41337.0}
"""
