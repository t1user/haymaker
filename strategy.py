from logbook import Logger

from streamers import VolumeStreamer
from candle import BreakoutCandle, RsiCandle, RepeatBreakoutCandle
from params import contracts


log = Logger(__name__)


candles = [BreakoutCandle(VolumeStreamer(params.volume,
                                         params.avg_periods),
                          **params.__dict__)
           for params in contracts]
