from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Type

from .base import Atom
from .misc import Action, Signal, sign
from .state_machine import StateMachine

log = logging.getLogger(__name__)


class AbstractBaseBinarySignalProcessor(Atom, ABC):
    """
    Process binary signals, i.e. long/short/off, as opposed to
    descrete signals, where signal strength is meaningful (e.g. signal
    assuming values -10...10 based on strength of conviction).

    Actual position size or even whether the position should be taken
    at all is not determined here, it's the job of `Portfolio`.

    Actual meaning of signals is defined in sub-classes, by overriding
    methods: :meth:`process_position` and :meth:`process_no_position`.

    Whatever the meaning of the signal coming in, signal coming out
    means strategy wants to take `action` in the direction of
    `signal`, as indicated by keys `action` and `signal` in the
    emitted dict.  Incoming signals that don't require any action will
    be stopped here and not propagated down the chain.

    In sub-class names `blip` means zero signal should be ignored,
    othewise absence of signal means there should be no position.

    Args:
    -----

    signal_fields - if `str`, single field of this name is used as
    signal in (open position) and signal out (close position), if
    `tuple` then first element is signal in and second is signal out

    state_machine - this is for testing only and should not be passed
    in non-testing environment
    """

    def __init__(
        self,
        signal_fields: str | tuple[str, str] | list[str] = "signal",
        state_machine: StateMachine | None = None,
    ) -> None:
        # passing state machine is only for testing
        if state_machine:
            self.sm = state_machine  # type: ignore

        if isinstance(signal_fields, str):
            self.signal_in_field = self.signal_out_field = signal_fields
        elif isinstance(signal_fields, (tuple, list)):
            if len(signal_fields) != 2:
                log.exception(
                    f"signal_fields must be a str or a 2 element tuple/list, "
                    f"not {signal_fields}"
                )
                raise
            else:
                self.signal_in_field = signal_fields[0]
                self.signal_out_field = signal_fields[1]

        self.strategy: str = ""
        self._position: float = 0.0
        Atom.__init__(self)

    def onData(self, data: dict[str, Any], *args) -> None:
        try:
            signal_in = data[self.signal_in_field]
            signal_out = data[self.signal_out_field]
        except KeyError:
            if self.signal_in_field == self.signal_out_field:
                fields = self.signal_in_field
            else:
                fields = f"{self.signal_in_field}, {self.signal_out_field}"
            log.exception(f"Missing `{fields}` expected by {self} in `onData`")
            return
        # 'strategy' must exist, likely set by onStart
        strategy = data.get("strategy") or self.strategy
        if result := self.process_signal(strategy, signal_in, signal_out):
            data.update(
                {
                    "action": result,
                    "target_position": self.target_position(signal_in, result),
                    "existing_position": self._position,
                }
            )
            self.dataEvent.emit(data)

    def target_position(self, signal, result):
        if result == "OPEN":
            return signal
        elif result == "REVERSE":
            return signal
        elif result == "CLOSE":
            return 0
        else:
            log.error(
                f"{self} generated unknown signal: {result} for strategy: "
                f"{self.strategy}"
            )

    def process_signal(
        self, strategy: str, signal_in: Signal, signal_out: Signal
    ) -> Action | None:
        if not self.position(strategy):
            return self.process_no_position(strategy, signal_in)
        elif not self.same_direction(strategy, signal_out):
            return self.process_position(strategy, signal_out)
        else:
            return None

    def position(self, strategy: str) -> Signal:
        """
        Which side of the market is position on: (short: -1, long: 1,
        no position: 0)
        """
        # record position state to assure that one value consistently used during
        # processing of the one data point
        self._position = sign(self.sm.position_and_order_for_strategy(strategy))
        return self._position

    def same_direction(self, strategy: str, signal: Signal) -> bool:
        """Is signal and position in the same direction?"""
        return self._position == signal

    def process_position(self, strategy: str, signal: Signal) -> Action | None:
        if signal == 0:
            return self.process_zero_signal_position(strategy, signal)
        else:
            return self.process_non_zero_signal_position(strategy, signal)

    def process_no_position(self, strategy: str, signal: Signal) -> Action | None:
        if signal == 0:
            return self.process_zero_signal_no_position(strategy, signal)
        else:
            return self.process_non_zero_signal_no_position(strategy, signal)

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def process_zero_signal_no_position(self, strategy, signal) -> Action | None:
        return None

    @abstractmethod
    def process_zero_signal_position(self, strategy, signal) -> Action | None: ...

    @abstractmethod
    def process_non_zero_signal_position(self, strategy, signal) -> Action | None: ...

    @abstractmethod
    def process_non_zero_signal_no_position(
        self, strategy, signal
    ) -> Action | None: ...


class BinarySignalProcessor(AbstractBaseBinarySignalProcessor):
    """
    * Zero signal means close position if position exists

    * Non-zero signal means:

    ** open new position if there is no position for the strategy

    ** ignore signal if it's in the same direction as existing
    position

    ** close position if the signal is in the direction opposite to
    existing position

    """

    def process_zero_signal_position(self, strategy, signal) -> Action | None:
        return "CLOSE"

    def process_non_zero_signal_position(self, strategy, signal) -> Action | None:
        # We've already checked signal is not same direction as position
        return "CLOSE"

    def process_non_zero_signal_no_position(self, strategy, signal) -> Action | None:
        return "OPEN"


class BlipBinarySignalProcessor(BinarySignalProcessor):
    """
    * Zero signal means do nothing

    * Non-zero signal means:

    ** open new position if there is no position for the strategy

    ** ignore signal if it's in the same direction as existing
    position

    ** close position if the signal is in the direction opposite to
    existing position

    """

    def process_zero_signal_position(self, strategy, signal) -> Action | None:
        return None


class LockableBinarySignalProcessor(AbstractBaseBinarySignalProcessor):
    """
    * Signals in the direction of last position are ignored (one side
    of the market is 'locked').  It's up to :class:`StateMachine` to
    determine which side is 'locked' based on position actually taken
    in the market (not just previously generated signals).

    * Zero signal means close position if position exists

    * Non-zero signal means:

    ** open new position if there is no position for the strategy

    ** ignore signal if it's in the same direction as existing
    position

    ** close position if the signal is in the direction opposite to
    existing position

    """

    def __init__(
        self,
        signal_fields: str | tuple[str, str] | list[str] = "signal",
        state_machine: StateMachine | None = None,
    ) -> None:
        self._lock_direction = 0
        self._lock = False
        super().__init__(signal_fields, state_machine=state_machine)

    def onData(self, data: dict[str, Any], *args) -> None:
        data.update({"lock_direction": self._lock_direction, "lock": self._lock})
        super().onData(data, *args)

    def locked(self, strategy: str, signal: Signal) -> bool:
        self._lock_direction = self.sm.locked(strategy)
        self._lock = self._lock_direction == signal
        return self._lock

    def process_zero_signal_position(self, strategy, signal) -> Action | None:
        return "CLOSE"

    def process_non_zero_signal_position(self, strategy, signal) -> Action | None:
        # We've already checked signal is not same direction as position
        # Zero signal means "CLOSE", oppposite signal means "REVERSE"
        return "CLOSE"

    def process_non_zero_signal_no_position(self, strategy, signal) -> Action | None:
        if self.locked(strategy, signal):
            return None
        else:
            return "OPEN"


class LockableBlipBinarySignalProcessor(LockableBinarySignalProcessor):
    """
    * Signals in the direction of last position are ignored (one side
    of the market is 'locked').  It's up to :class:`StateMachine` to
    determine which side is 'locked' based on position actually taken
    in the market (not just previously generated signals).

    * Zero signal means do nothing

    * Non-zero signal means:

    ** open new position if there is no position for the strategy

    ** ignore signal if it's in the same direction as existing
    position

    ** close position if the signal is in the direction opposite to
    existing position

    """

    def process_zero_signal_position(self, strategy, signal) -> Action | None:
        return None


class AlwaysOnLockableBinarySignalProcessor(LockableBinarySignalProcessor):
    """
    * Signals in the direction of last position are ignored (one side
    of the market is 'locked').  It's up to :class:`StateMachine` to
    determine which side is 'locked' based on position actually taken
    in the market (not just previously generated signals).

    * Zero signal means close position if position exists

    * Non-zero signal means:

    ** open new position if there is no position for the strategy

    ** ignore signal if it's in the same direction as existing
    position

    ** reverse position if the signal is in the direction opposite to
    existing position
    """

    def __init__(
        self,
        signal_fields: str | tuple[str, str] | list[str] = "signal",
        state_machine: StateMachine | None = None,
    ) -> None:
        super().__init__(signal_fields, state_machine)
        if self.signal_in_field != self.signal_out_field:
            log.exception(f"{self} requires single signal field, not {signal_fields}")
            raise

    def process_non_zero_signal_position(self, strategy, signal):
        return "REVERSE"


class AlwaysOnBinarySignalProcessor(BinarySignalProcessor):
    """
    * Zero signal means close position if position exists

    * Non-zero signal means:

    ** open new position if there is no position for the strategy

    ** ignore signal if it's in the same direction as existing
    position

    ** close position if the signal is in the direction opposite to
    existing position

    """

    def __init__(
        self,
        signal_fields: str | tuple[str, str] | list[str] = "signal",
        state_machine: StateMachine | None = None,
    ) -> None:
        super().__init__(signal_fields, state_machine)
        if self.signal_in_field != self.signal_out_field:
            log.exception(f"{self} requires single signal field, not {signal_fields}")
            raise

    def process_non_zero_signal_position(self, strategy, signal) -> Action | None:
        return "REVERSE"


def binary_signal_processor_factory(
    lockable=False, always_on=False
) -> Type[AbstractBaseBinarySignalProcessor]:
    """
    Helper function to return appropriate class based on parameters.

    Parameters:

        lockable:
            True - no signals in the direction of last position if the last
            position was stopped out (allowed, if position was closed through
            means other than stop-loss)

        always_on:
            True - 'CLOSE' signal also opens position in reverese direction
    """
    if lockable and always_on:
        return AlwaysOnLockableBinarySignalProcessor
    elif lockable:
        return LockableBinarySignalProcessor
    elif always_on:
        return AlwaysOnBinarySignalProcessor
    else:
        return BinarySignalProcessor
