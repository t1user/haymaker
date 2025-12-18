from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import ib_insync as ibi


def is_active(
    time_tuples: list[tuple[datetime, datetime]] | None = None,
    now: datetime | None = None,
) -> bool:
    """
    Given list of trading hours tuples from `.process_trading_hours`
    check if market is active at the moment.
    """
    if not time_tuples:
        return False

    if not now:
        now = datetime.now(tz=timezone.utc)

    def test_p(t):
        return t[0] < now < t[1]

    for t in time_tuples:
        if test_p(t):
            return True
    return False


def next_active(
    time_tuples: list[tuple[datetime, datetime]] | None = None,
    now: datetime | None = None,
) -> datetime | None:
    """
    Given list of trading hours tuples from `.process_trading_hours`
    return time of nearest market re-open (regardless if market is
    open now).  Should be used after it has been tested that
    `.is_active` is False.
    """

    if not now:
        now = datetime.now(tz=ZoneInfo("UTC"))

    if not time_tuples:
        return None

    left_edges = [e[0] for e in time_tuples if e[0] > now]
    # print(left_edges)
    return left_edges[0]


def process_trading_hours(
    th: str, input_tz: str = "US/Central", *, output_tz: str = "UTC"
) -> list[tuple[datetime, datetime]]:
    """
    Given string from :attr:`ibi.ContractDetails.tradingHours` return
    active hours as a list of (from, to) tuples.

    Args:
    -----

    tzname: instrument's timezone

    output_tzname: output will be converted to this timezone (best if
    left at UTC); this param is really for testing
    """
    try:
        input_tz_: ZoneInfo | None = ZoneInfo(input_tz)
        output_tz_: ZoneInfo | None = ZoneInfo(output_tz)
    except ValueError:
        input_tz_ = None
        output_tz_ = None

    def datetime_tuples(s: str) -> tuple[datetime | None, datetime | None]:
        def to_datetime(datetime_string: str) -> datetime | None:
            if datetime_string[-6:] == "CLOSED":
                return None
            else:
                return (
                    datetime.strptime(datetime_string, "%Y%m%d:%H%M")
                    .replace(tzinfo=input_tz_)
                    .astimezone(tz=output_tz_)
                )

        try:
            f, t = s.split("-")
        except ValueError:
            return (None, None)

        return to_datetime(f), to_datetime(t)

    out = []
    for i in th.split(";"):
        tuples = datetime_tuples(i)
        if not tuples[0]:
            continue
        else:
            out.append(tuples)
    return out  # type: ignore


def typical_session_length(
    time_tuples: list[tuple[datetime, datetime]],
) -> timedelta:
    time_deltas = [t[1] - t[0] for t in time_tuples]
    # this is mode calculation
    return max(set(time_deltas), key=time_deltas.count)


@dataclass
class Details:
    """
    Wrapper object for :class:`ib_insync.contract.ContractDetails` extracting
    and processing information that's most relevant for Haymaker.

    Attributes:
        details (ibi.ContractDetails): The original contract details object.
        trading_hours (list[tuple[datetime, datetime]]): List of tuples with
            start and end of trading hours for this contract.
        liquid_hours (list[tuple[datetime, datetime]]): List of tuples with
            start and end of liquid hours for this contract.
    """

    _fields: ClassVar[list[str]] = [f.name for f in fields(ibi.ContractDetails)]
    details: ibi.ContractDetails
    trading_hours: list[tuple[datetime, datetime]] = field(init=False, repr=False)
    liquid_hours: list[tuple[datetime, datetime]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.trading_hours = self._process_trading_hours(
            self.details.tradingHours, self.details.timeZoneId
        )
        self.liquid_hours = self._process_trading_hours(
            self.details.liquidHours, self.details.timeZoneId
        )

    def __getattr__(self, name: str) -> Any:
        if name in self._fields:
            return getattr(self.details, name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def is_open(self, _now: datetime | None = None) -> bool:
        """
        Given current time check if the market is open for underlying contract.

        Args:
            _now (Optional[datetime], optional): Defaults to None.
            If not provided current time will be used. Only situation when
            it's useful to provide `_now` is in testing.

        Returns:
            bool: True if market is open, False otherwise.
        """
        return self._is_active(self.trading_hours, _now)

    def is_liquid(self, _now: datetime | None = None) -> bool:
        """
        Given current time check if the market is during liquid hours
        for underlying contract.

        Args:
            _now (Optional[datetime], optional): . Defaults to None.
            If not provided current time will be used. Only situation when
            it's useful to provide `_now` is in testing.

        Returns:
            bool: True if market is liquid, False otherwise.
        """
        return self._is_active(self.liquid_hours, _now)

    def next_open(self, _now: datetime | None = None) -> datetime | None:
        """
        Return time of nearest market re-open (regardless if market is
        open now).  Should be used after it has been tested that
        :meth:`is_active` is False.

        Args:
            _now (Optional[datetime], optional): Defaults to None.
            If not provided current time will be used. Only situation when
            it's useful to provide `_now` is in testing.
        """
        return self._next_open(self.trading_hours, _now)

    _process_trading_hours = staticmethod(process_trading_hours)
    _is_active = staticmethod(is_active)
    _next_open = staticmethod(next_active)
