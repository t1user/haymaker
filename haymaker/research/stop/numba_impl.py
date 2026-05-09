from typing import Tuple

import numpy as np
from numba import jit  # type: ignore

from .python_impl import (
    BaseBracket,
    FixedStop,
    NoAdjust,
    NoTakeProfit,
    NoTimeStop,
    StopParams,
    TrailStop,
)

STOP_FIXED = 0
STOP_TRAIL = 1
RuntimeConfig = Tuple[int, float, int, bool, int, float, float]


def _stop_mode_code(stop_type: type[BaseBracket]) -> int:
    if stop_type is FixedStop:
        return STOP_FIXED
    if stop_type is TrailStop:
        return STOP_TRAIL
    raise ValueError(f"Unsupported stop type: {stop_type}")


def runtime_config(params: StopParams) -> RuntimeConfig:
    stop, tp, ts, adjust = params

    stop_mode = _stop_mode_code(stop)
    tp_multiple = 0.0 if tp is NoTakeProfit else float(getattr(tp, "multiple"))
    time_stop = 0 if ts is NoTimeStop else int(ts.periods)
    adjust_enabled = adjust is not NoAdjust
    adjust_mode = STOP_FIXED
    adjust_trigger_multiple = 0.0
    adjust_stop_multiple = 0.0

    if adjust_enabled:
        adjust_mode = _stop_mode_code(adjust.adjusted_stop)
        adjust_trigger_multiple = float(adjust.trigger_multiple)
        adjust_stop_multiple = float(adjust.stop_multiple)

    return (
        stop_mode,
        tp_multiple,
        time_stop,
        adjust_enabled,
        adjust_mode,
        adjust_trigger_multiple,
        adjust_stop_multiple,
    )


@jit(nopython=True)
def _evaluate_stop_bracket(
    stop_mode: int,
    position: int,
    stop_trigger: float,
    stop_distance: float,
    high: float,
    low: float,
) -> Tuple[float, float]:
    triggered = 0.0

    if stop_mode == STOP_TRAIL:
        if position == -1:
            if stop_trigger <= high:
                triggered = stop_trigger
            else:
                candidate = low + stop_distance
                if candidate < stop_trigger:
                    stop_trigger = candidate
        else:
            if stop_trigger >= low:
                triggered = stop_trigger
            else:
                candidate = high - stop_distance
                if candidate > stop_trigger:
                    stop_trigger = candidate
    else:
        if position == 1 and stop_trigger >= low:
            triggered = stop_trigger
        elif position == -1 and stop_trigger <= high:
            triggered = stop_trigger

    return triggered, stop_trigger


@jit(nopython=True)
def _evaluate_take_profit(
    position: int, tp_trigger: float, high: float, low: float
) -> float:
    if position == -1 and tp_trigger >= low:
        return tp_trigger
    if position == 1 and tp_trigger <= high:
        return tp_trigger
    return 0.0


@jit(nopython=True)
def _evaluate_time_stop(counter: int, periods: int, price: float) -> Tuple[float, int]:
    counter += 1
    if counter >= periods:
        return price, counter
    return 0.0, counter


@jit(nopython=True)
def _evaluate_brackets(
    position: int,
    high: float,
    low: float,
    price: float,
    stop_mode: int,
    stop_distance: float,
    stop_trigger: float,
    tp_multiple: float,
    tp_trigger: float,
    time_stop: int,
    ts_counter: int,
) -> Tuple[int, float, int, float]:
    triggered, stop_trigger = _evaluate_stop_bracket(
        stop_mode, position, stop_trigger, stop_distance, high, low
    )
    if triggered != 0.0:
        return 0, stop_trigger, ts_counter, triggered * -position

    if tp_multiple:
        triggered = _evaluate_take_profit(position, tp_trigger, high, low)
        if triggered != 0.0:
            return 0, stop_trigger, ts_counter, triggered * -position

    if time_stop:
        triggered, ts_counter = _evaluate_time_stop(ts_counter, time_stop, price)
        if triggered != 0.0:
            return 0, stop_trigger, ts_counter, triggered * -position

    return position, stop_trigger, ts_counter, 0.0


@jit(nopython=True)
def _adjust_stop_state(
    position: int,
    high: float,
    low: float,
    stop_mode: int,
    stop_distance: float,
    stop_trigger: float,
    adjust_done: bool,
    adjust_mode: int,
    adjust_trigger: float,
    adjusted_stop_distance: float,
) -> Tuple[int, float, float, bool]:
    if adjust_done:
        return stop_mode, stop_distance, stop_trigger, adjust_done

    if (position == -1 and adjust_trigger >= low) or (
        position == 1 and adjust_trigger <= high
    ):
        adjusted_trigger = adjust_trigger - adjusted_stop_distance * position
        return adjust_mode, adjusted_stop_distance, adjusted_trigger, True

    return stop_mode, stop_distance, stop_trigger, adjust_done


@jit(nopython=True)
def _stop_loss_numba(
    first: np.ndarray,
    second: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    distance_values: np.ndarray,
    scalar_distance: float,
    distance_is_scalar: bool,
    price_values: np.ndarray,
    scheduled_close: np.ndarray,
    use_blip: bool,
    stop_mode: int,
    tp_multiple: float,
    time_stop: int,
    adjust_enabled: bool,
    adjust_mode: int,
    adjust_trigger_multiple: float,
    adjust_stop_multiple: float,
) -> np.ndarray:
    out = np.zeros((len(first), 4), dtype=np.float64)

    position = 0
    stop_mode_state = stop_mode
    stop_distance = 0.0
    stop_trigger = 0.0
    tp_trigger = 0.0
    ts_counter = 0
    adjust_done = True
    adjust_trigger = 0.0
    adjusted_stop_distance = 0.0

    for i in range(len(first)):
        first_value = int(first[i])
        second_value = int(second[i])
        high = high_values[i]
        low = low_values[i]
        distance = scalar_distance if distance_is_scalar else distance_values[i]
        price = price_values[i]
        force_close = scheduled_close[i]

        open_price = 0.0
        close_price = 0.0
        stop_price = 0.0
        blip = 0
        close_blip = 0
        target_position = 0
        transaction = 0

        if use_blip:
            blip = first_value
            close_blip = second_value
            if blip and not force_close:
                assert distance > 0
        else:
            target_position = first_value
            transaction = second_value
            if transaction and not force_close:
                assert distance > 0

        if force_close:
            if position:
                close_price = price * -position
                position = 0
        elif position:
            should_close = (
                close_blip == -position if use_blip else transaction == -position
            )
            if should_close:
                close_price = price * -position
                position = 0

                if not use_blip and transaction and transaction == target_position:
                    stop_mode_state = stop_mode
                    stop_distance = distance
                    stop_trigger = price - distance * transaction
                    tp_trigger = price + distance * transaction * tp_multiple
                    ts_counter = 0
                    adjust_done = not adjust_enabled
                    adjust_trigger = (
                        price + distance * adjust_trigger_multiple * transaction
                    )
                    adjusted_stop_distance = distance * adjust_stop_multiple
                    open_price = price * transaction
                    position = transaction
                    position, stop_trigger, ts_counter, stop_price = _evaluate_brackets(
                        position,
                        high,
                        low,
                        price,
                        stop_mode_state,
                        stop_distance,
                        stop_trigger,
                        tp_multiple,
                        tp_trigger,
                        time_stop,
                        ts_counter,
                    )
            else:
                position, stop_trigger, ts_counter, stop_price = _evaluate_brackets(
                    position,
                    high,
                    low,
                    price,
                    stop_mode_state,
                    stop_distance,
                    stop_trigger,
                    tp_multiple,
                    tp_trigger,
                    time_stop,
                    ts_counter,
                )
                if position and adjust_enabled:
                    (
                        stop_mode_state,
                        stop_distance,
                        stop_trigger,
                        adjust_done,
                    ) = _adjust_stop_state(
                        position,
                        high,
                        low,
                        stop_mode_state,
                        stop_distance,
                        stop_trigger,
                        adjust_done,
                        adjust_mode,
                        adjust_trigger,
                        adjusted_stop_distance,
                    )
        else:
            should_open = (
                blip != 0
                if use_blip
                else (transaction != 0 and transaction == target_position)
            )
            if should_open:
                transaction = blip if use_blip else transaction
                stop_mode_state = stop_mode
                stop_distance = distance
                stop_trigger = price - distance * transaction
                tp_trigger = price + distance * transaction * tp_multiple
                ts_counter = 0
                adjust_done = not adjust_enabled
                adjust_trigger = (
                    price + distance * adjust_trigger_multiple * transaction
                )
                adjusted_stop_distance = distance * adjust_stop_multiple
                open_price = price * transaction
                position = transaction
                position, stop_trigger, ts_counter, stop_price = _evaluate_brackets(
                    position,
                    high,
                    low,
                    price,
                    stop_mode_state,
                    stop_distance,
                    stop_trigger,
                    tp_multiple,
                    tp_trigger,
                    time_stop,
                    ts_counter,
                )

        out[i, 0] = position
        out[i, 1] = open_price
        out[i, 2] = close_price
        out[i, 3] = stop_price

    return out


def run_stop_loss(
    first: np.ndarray,
    second: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    distance: float | np.ndarray,
    price: np.ndarray,
    scheduled_close: np.ndarray,
    use_blip: bool,
    params: StopParams,
) -> np.ndarray:
    if isinstance(distance, np.ndarray):
        distance_values = distance
        scalar_distance = 0.0
        distance_is_scalar = False
    else:
        distance_values = np.empty(0, dtype=np.float64)
        scalar_distance = float(distance)
        distance_is_scalar = True

    return _stop_loss_numba(
        first,
        second,
        high,
        low,
        distance_values,
        scalar_distance,
        distance_is_scalar,
        price,
        scheduled_close,
        use_blip,
        *runtime_config(params),
    )
