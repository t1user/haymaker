import datetime
import zoneinfo

from ib_insync import BarData

sample_barDataList = [
    BarData(
        date=datetime.datetime(
            2026, 1, 28, 17, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7021.5,
        high=7022.0,
        low=6994.5,
        close=6996.0,
        volume=19047.0,
        average=7006.375,
        barCount=7712,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 28, 18, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=6996.0,
        high=6999.25,
        low=6985.75,
        close=6999.25,
        volume=16297.0,
        average=6993.45,
        barCount=7407,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 28, 19, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=6999.5,
        high=6999.75,
        low=6987.0,
        close=6994.75,
        volume=14055.0,
        average=6993.825,
        barCount=6585,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 28, 20, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=6994.75,
        high=7006.25,
        low=6993.5,
        close=7003.0,
        volume=11900.0,
        average=7001.475,
        barCount=5077,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 28, 21, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7003.0,
        high=7009.75,
        low=7003.0,
        close=7008.75,
        volume=6444.0,
        average=7006.875,
        barCount=2895,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 28, 22, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7008.75,
        high=7016.0,
        low=7008.0,
        close=7015.5,
        volume=5115.0,
        average=7011.9,
        barCount=2279,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 28, 23, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7015.5,
        high=7019.75,
        low=7012.25,
        close=7017.75,
        volume=4811.0,
        average=7016.525,
        barCount=2341,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7017.75,
        high=7023.75,
        low=7017.25,
        close=7020.0,
        volume=6132.0,
        average=7020.7,
        barCount=2509,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 1, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7019.75,
        high=7029.5,
        low=7017.5,
        close=7026.5,
        volume=10037.0,
        average=7024.55,
        barCount=4003,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 2, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7026.5,
        high=7029.5,
        low=7019.0,
        close=7024.75,
        volume=17590.0,
        average=7025.2,
        barCount=7447,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 3, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7024.75,
        high=7029.25,
        low=7018.75,
        close=7019.25,
        volume=12190.0,
        average=7025.975,
        barCount=5069,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 4, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7019.0,
        high=7023.5,
        low=7012.75,
        close=7013.75,
        volume=10547.0,
        average=7019.275,
        barCount=4458,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 5, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7014.0,
        high=7022.0,
        low=7010.0,
        close=7019.5,
        volume=14351.0,
        average=7017.075,
        barCount=5951,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 6, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7019.5,
        high=7024.0,
        low=7013.25,
        close=7019.0,
        volume=13085.0,
        average=7019.65,
        barCount=5292,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 7, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7018.75,
        high=7026.75,
        low=7015.0,
        close=7023.5,
        volume=17902.0,
        average=7021.075,
        barCount=6896,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 8, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=7023.5,
        high=7023.75,
        low=6965.5,
        close=6966.5,
        volume=239685.0,
        average=6994.9,
        barCount=92119,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 9, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=6966.5,
        high=6967.25,
        low=6902.25,
        close=6910.75,
        volume=457142.0,
        average=6936.575,
        barCount=166764,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 10, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=6911.0,
        high=6940.75,
        low=6898.25,
        close=6937.5,
        volume=289509.0,
        average=6925.1,
        barCount=111874,
    ),
    BarData(
        date=datetime.datetime(
            2026, 1, 29, 11, 0, tzinfo=zoneinfo.ZoneInfo(key="US/Central")
        ),
        open=6937.25,
        high=6967.75,
        low=6936.75,
        close=6942.5,
        volume=165169.0,
        average=6954.4,
        barCount=59129,
    ),
]
