import pickle
from functools import partial
from utils import breakout_strategy, bootstrap, m_proc
import sys
sys.path.append('/home/tomek/ib_tools/')
from datastore_pytables import Store  # noqa


store = Store()

contract = store.read('/cont/min/NQ_20191220_GLOBEX_USD').sort_index()


table = bootstrap(contract,
                  start='20180701',
                  end='20181231',
                  paths=100)


func = partial(breakout_strategy,
               time_int=30,
               periods=[5, 10, 20, 40, 80, 160, ],
               ema_fast=10,
               ema_slow=120,
               atr_periods=80,
               sl_atr=1,
               )


results = m_proc(table, func)
print(results)

with open('bootstrap_paths.pickle', 'wb') as f:
    pickle.dump(results, f)
