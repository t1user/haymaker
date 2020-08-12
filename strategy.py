from logbook import Logger

from streamers import VolumeStreamer
from candle import BreakoutCandle, RsiCandle, RepeatBreakoutCandle
from portfolio import AdjustedPortfolio
from params import contracts


log = Logger(__name__)


candles = [BreakoutCandle(VolumeStreamer(params.volume,
                                         params.avg_periods),
                          contract_fields=['contract', 'micro_contract'],
                          **params.__dict__)
           for params in contracts]
portfolio = AdjustedPortfolio(target_vol=.5)

strategy_kwargs = {'candles': candles,
                   'portfolio': portfolio}
