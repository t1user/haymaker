from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Type

import numpy as np

StopMode = Literal["fixed", "trail"]
StopParams = Tuple[
    Type["BaseBracket"],
    Type["BaseBracket"],
    Type["TimeStop"],
    Type["Adjust"],
]


class BaseBracket(ABC):
    def __init__(self, distance: float, transaction: int, entry: float = 0) -> None:
        self.distance = distance
        self.position = transaction
        self.entry = entry
        self.trigger = self.set_trigger()

    @abstractmethod
    def evaluate(self, high: float, low: float, price: float = 0.0) -> float:
        """
        Bracket has been triggered?
        True: return price to execute transaction
        False: return 0
        """

    def set_trigger(self, *args: Any) -> float:
        """
        This is for stop loss. For take profit the method has to be overridden.
        """
        return self.entry - self.distance * self.position

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            + "("
            + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
            + ")"
        )


class TrailStop(BaseBracket):
    def evaluate(self, high: float, low: float, price: float = 0.0) -> float:
        if self.position == -1:
            if self.trigger <= high:
                return self.trigger
            self.trigger = min(self.trigger, low + self.distance)
            return 0.0
        if self.position == 1:
            if self.trigger >= low:
                return self.trigger
            self.trigger = max(self.trigger, high - self.distance)
            return 0.0
        raise ValueError(
            f"Evaluating TrailStop for zero position. Position: {self.position}"
        )


class FixedStop(BaseBracket):
    def evaluate(self, high: float, low: float, price: float = 0.0) -> float:
        if self.position == 1 and self.trigger >= low:
            return self.trigger
        if self.position == -1 and self.trigger <= high:
            return self.trigger
        return 0.0


class TakeProfit(BaseBracket):
    multiple: float

    @classmethod
    def from_args(cls, multiple: float) -> Type["TakeProfit"]:
        if multiple:
            cls.multiple = multiple
            return cls
        return NoTakeProfit

    def evaluate(self, high: float, low: float, price: float = 0.0) -> float:
        if self.position == -1 and self.trigger >= low:
            return self.trigger
        if self.position == 1 and self.trigger <= high:
            return self.trigger
        return 0.0

    def set_trigger(self, *args: Any) -> float:
        return self.entry + self.distance * self.position * self.multiple


class NoTakeProfit(TakeProfit):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def evaluate(self, high: float, low: float, price: float = 0.0) -> float:
        return 0.0


class TimeStop:
    periods: int = 0

    @classmethod
    def from_args(cls, periods: int) -> type["TimeStop"]:
        if periods:
            cls.periods = periods
            return cls
        return NoTimeStop

    def __init__(self) -> None:
        self.counter = 0

    def evaluate(self, high: float, low: float, price: float = 0.0) -> float:
        """
        Interface in line with BaseBracket protocol.

        Bracket has been triggered?
        True: return price to execute transaction
        False: return 0
        """
        self.counter += 1
        return float((self.counter >= self.periods) * price)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            + "("
            + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
            + ")"
        )


class NoTimeStop(TimeStop):
    def evaluate(self, high: float, low: float, price: float = 0.0) -> float:
        return 0.0


stop_dict: Dict[StopMode, Type[BaseBracket]] = {
    "fixed": FixedStop,
    "trail": TrailStop,
}


class Adjust:
    adjusted_stop: Type[BaseBracket]
    trigger_multiple: float
    stop_multiple: float

    @classmethod
    def from_args(
        cls,
        adjust_tuple: Optional[Tuple[StopMode, float, float]] = None,
    ) -> type["Adjust"]:
        if adjust_tuple:
            adjusted_stop = stop_dict.get(adjust_tuple[0])
            if adjusted_stop is None:
                raise ValueError(
                    f"Invalid adjusted stop loss type: {adjust_tuple[0]}. "
                    "Must be 'fixed' or 'trail'"
                )
            cls.adjusted_stop = adjusted_stop
            cls.trigger_multiple = adjust_tuple[1]
            cls.stop_multiple = adjust_tuple[2]
            return cls
        return NoAdjust

    def __init__(
        self, stop_distance: float, transaction: int, entry: float = 0
    ) -> None:
        self.entry = entry
        self.adjusted_stop_distance = stop_distance * self.stop_multiple
        self.adjusted_trigger_distance = stop_distance * self.trigger_multiple
        self.position = transaction
        self.set_trigger()
        self.done = False

    def evaluate(
        self, order: BaseBracket, high: float, low: float, price: float = 0.0
    ) -> BaseBracket:
        """
        Verify whether stop should be adjusted, return correct stop
        (adjusted or not).
        """
        if self.done:
            return order
        if (self.position == -1 and self.trigger >= low) or (
            self.position == 1 and self.trigger <= high
        ):
            adjusted = self.adjusted_stop(
                self.adjusted_stop_distance, order.position, entry=self.trigger
            )
            self.done = True
            return adjusted
        return order

    def set_trigger(self) -> None:
        self.trigger = self.entry + self.adjusted_trigger_distance * self.position

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            + "("
            + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
            + ")"
        )


class NoAdjust(Adjust):
    def __init__(self, *args: Any) -> None:
        pass

    def evaluate(
        self,
        order: BaseBracket,
        high: float = 0.0,
        low: float = 0.0,
        price: float = 0.0,
    ) -> BaseBracket:
        return order


class Context:
    stop: BaseBracket
    tp: BaseBracket
    ts: TimeStop
    adjust: Adjust

    def __init__(
        self,
        stop: Type[BaseBracket],
        tp: Type[BaseBracket],
        ts: Type[TimeStop],
        adjust: Type[Adjust],
    ) -> None:
        self._stop = stop
        self._tp = tp
        self._ts = ts
        self._adjust = adjust
        self.position = 0

    def __call__(self, row: np.ndarray) -> Tuple[int, float, float, float]:
        (
            self.target_position,
            self.transaction,
            self.high,
            self.low,
            self.distance,
            self.price,
        ) = row

        if self.transaction:
            assert self.distance > 0, (
                f"Wrong value for stop loss distance: {self.distance}"
            )

        return self.dispatch()

    def process_values(
        self,
        first: int,
        second: int,
        high: float,
        low: float,
        distance: float,
        price: float,
    ) -> Tuple[int, float, float, float]:
        self.target_position = first
        self.transaction = second
        self.high = high
        self.low = low
        self.distance = distance
        self.price = price

        if self.transaction:
            assert self.distance > 0, (
                f"Wrong value for stop loss distance: {self.distance}"
            )

        return self.dispatch()

    def dispatch(self) -> Tuple[int, float, float, float]:
        self.open_price = 0.0
        self.close_price = 0.0
        self.stop_price = 0.0

        if self.position:
            self.eval_for_close()
        else:
            self.eval_for_open()
        return (
            self.position,
            self.open_price,
            self.close_price,
            self.stop_price,
        )

    def eval_for_close(self) -> None:
        if self.transaction == -self.position:
            self.close_position()
            self.eval_for_open()
        else:
            self.eval_brackets()
            self.eval_adjust()

    def eval_for_open(self) -> None:
        if self.transaction and self.transaction == self.target_position:
            self.open_position(self.transaction)

    def open_position(self, transaction: int) -> None:
        self.stop = self._stop(self.distance, transaction, self.price)
        self.tp = self._tp(self.distance, transaction, self.price)
        self.ts = self._ts()
        self.adjust = self._adjust(self.distance, transaction, self.price)
        self.open_price = self.price * transaction
        self.position = transaction
        self.eval_brackets()

    def close_position(self) -> None:
        self.close_price = self.price * -self.position
        self.position = 0

    def eval_brackets(self) -> None:
        for bracket in (self.stop, self.tp, self.ts):
            if p := bracket.evaluate(self.high, self.low, self.price):
                self.stop_price = p * -self.position
                self.position = 0
                if self.stop_price == 0:
                    raise ValueError(
                        "Position has been closed by a bracket "
                        "but no transaction price given"
                    )
                return

    def eval_adjust(self) -> None:
        self.stop = self.adjust.evaluate(self.stop, self.high, self.low)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            + "("
            + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
            + ")"
        )


class BlipContext(Context):
    def __call__(self, row: np.ndarray) -> Tuple[int, float, float, float]:
        (
            self.blip,
            self.close_blip,
            self.high,
            self.low,
            self.distance,
            self.price,
        ) = row

        if self.blip:
            assert self.distance > 0, (
                f"Wrong value for stop loss distance: {self.distance}"
            )

        return self.dispatch()

    def process_values(
        self,
        first: int,
        second: int,
        high: float,
        low: float,
        distance: float,
        price: float,
    ) -> Tuple[int, float, float, float]:
        self.blip = first
        self.close_blip = second
        self.high = high
        self.low = low
        self.distance = distance
        self.price = price

        if self.blip:
            assert self.distance > 0, (
                f"Wrong value for stop loss distance: {self.distance}"
            )

        return self.dispatch()

    def eval_for_close(self) -> None:
        if self.close_blip == -self.position:
            self.close_position()
        else:
            self.eval_brackets()
            self.eval_adjust()

    def eval_for_open(self) -> None:
        if self.blip:
            self.open_position(self.blip)


def _stop_loss(data: np.ndarray, stop: Context) -> np.ndarray:
    """
    Args:
    -----

    data: columns have the meaning required by the context being passed.
    """
    position = np.zeros((data.shape[0], 1), dtype=np.int8)
    open_price = np.zeros((data.shape[0], 1), dtype=np.float64)
    close_price = np.zeros((data.shape[0], 1), dtype=np.float64)
    stop_price = np.zeros((data.shape[0], 1), dtype=np.float64)

    for i, row in enumerate(data):
        position[i], open_price[i], close_price[i], stop_price[i] = stop(row)

    return np.concatenate(
        (position, open_price, close_price, stop_price), axis=1
    )


def run_stop_loss(
    first: np.ndarray,
    second: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    distance: float | np.ndarray,
    price: np.ndarray,
    use_blip: bool,
    params: StopParams,
) -> np.ndarray:
    context: Context = BlipContext(*params) if use_blip else Context(*params)
    out = np.zeros((len(first), 4), dtype=np.float64)

    if isinstance(distance, np.ndarray):
        for i in range(len(first)):
            out[i, :] = context.process_values(
                int(first[i]),
                int(second[i]),
                float(high[i]),
                float(low[i]),
                float(distance[i]),
                float(price[i]),
            )
    else:
        current_distance = float(distance)
        for i in range(len(first)):
            out[i, :] = context.process_values(
                int(first[i]),
                int(second[i]),
                float(high[i]),
                float(low[i]),
                current_distance,
                float(price[i]),
            )

    return out


def param_factory(
    mode: StopMode,
    tp_multiple: float = 0,
    time_stop: int = 0,
    adjust_tuple: Optional[Tuple[StopMode, float, float]] = None,
) -> StopParams:
    """
    Verify validity of parameters and based on them return appropriate
    objects for Context.

    Stop is required. Take profit and adjust if not used set to objects that
    don't do anything.
    """
    if tp_multiple and adjust_tuple:
        assert adjust_tuple[1] < tp_multiple, (
            "Take profit multiple must be > adjust trigger multiple. Otherwise"
            " position would be closed before stop loss can be adjusted."
        )

    stop = stop_dict.get(mode)
    if stop is None:
        raise ValueError(f"Invalid stop loss type: {mode}. Must be 'fixed' or 'trail'")

    tp = TakeProfit.from_args(tp_multiple)
    ts = TimeStop.from_args(time_stop)
    adjust = Adjust.from_args(adjust_tuple)

    return (stop, tp, ts, adjust)
