from trader import Params

nq = Params(
    contract=('NQ', 'GLOBEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=1,
    atr_periods=180,
    # avg_periods=60,
    volume=10000,
    alloc=0.3)

es = Params(
    contract=('ES', 'GLOBEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=120,
    ema_slow=320,
    sl_atr=3,
    atr_periods=180,
    # avg_periods=60,
    volume=33000,
    alloc=0.3)

gc = Params(
    contract=('GC', 'NYMEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=1,
    atr_periods=180,
    # avg_periods=60,
    volume=5400,
    alloc=0.35)

cl = Params(
    contract=('CL', 'NYMEX'),
    periods=[5, 10, 20, 40, 80, 160],
    ema_fast=5,
    ema_slow=120,
    sl_atr=2,
    atr_periods=180,
    # avg_periods=60,
    volume=11500,
    alloc=0.05)

contracts = [nq, es, gc, cl]
