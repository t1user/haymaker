from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Union

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

    * Zero signal means close position if position exists, ignored
    otherwise <<<<--- THIS IS WRONG

    * Non-zero signal means:

    ** open new position if there is no position for the strategy

    ** ignore signal if it's in the same direction as existing
    position

    ** reverse position if the signal is in the direction opposite to
    existing position

    This behaviour can be modified in sub-classes, by overriding
    methods: :meth:`process_position` and :meth:`process_no_position`.

    Whatever the meaning of the signal coming in, signal coming out
    means strategy wants to take `action` in the direction of
    `signal`, as indicated by keys `action` and `signal` in the
    emitted dict.  Incoming signals that don't require any action will
    be stopped here and not propagated down the chain.

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
        signal_fields: Union[str, tuple[str, str], list[str]] = "signal",
        state_machine: Optional[StateMachine] = None,
    ) -> None:
        # passing state machine is only for testing
        if state_machine:
            self.sm = state_machine

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

        # this is just definition of class members
        self.strategy: str = ""
        self._position: float = 0.0
        Atom.__init__(self)

    # onStart should set strategy
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
    ) -> Optional[Action]:
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
        self._position = sign(self.sm.strategy[strategy].position)
        return self._position

    def same_direction(self, strategy: str, signal: Signal) -> bool:
        """Is signal and position in the same direction?"""
        return self.position(strategy) == signal

    def process_position(self, strategy: str, signal: Signal) -> Optional[Action]:
        if signal == 0:
            return self.process_zero_signal_position(strategy, signal)
        else:
            return self.process_non_zero_signal_position(strategy, signal)

    def process_no_position(self, strategy: str, signal: Signal) -> Optional[Action]:
        if signal == 0:
            return self.process_zero_signal_no_position(strategy, signal)
        else:
            return self.process_non_zero_signal_no_position(strategy, signal)

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def process_zero_signal_no_position(self, strategy, signal) -> Optional[Action]:
        return None

    @abstractmethod
    def process_zero_signal_position(self, strategy, signal) -> Optional[Action]:
        ...

    @abstractmethod
    def process_non_zero_signal_position(self, strategy, signal) -> Optional[Action]:
        ...

    @abstractmethod
    def process_non_zero_signal_no_position(self, strategy, signal) -> Optional[Action]:
        ...


class BinarySignalProcessor(AbstractBaseBinarySignalProcessor):
    def process_zero_signal_position(self, strategy, signal) -> Optional[Action]:
        return None

    def process_non_zero_signal_position(self, strategy, signal) -> Optional[Action]:
        # We've already checked signal is not same direction as position
        return "CLOSE"

    def process_non_zero_signal_no_position(self, strategy, signal) -> Optional[Action]:
        return "OPEN"


class LockableBinarySignalProcessor(AbstractBaseBinarySignalProcessor):
    """
    * Signals in the direction of last position are ignored (one side
    of the market is 'locked').  It's up to :class:`StateMachine` to
    determine which side is 'locked' based on position actually taken
    in the market (not just previously generated signals).

    * Zero signal means close position if position exists, ignored
    otherwise
    """

    def __init__(
        self,
        signal_fields: Union[str, tuple[str, str], list[str]] = "signal",
        state_machine: Optional[StateMachine] = None,
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

    def process_zero_signal_position(self, strategy, signal) -> Optional[Action]:
        return "CLOSE"

    def process_non_zero_signal_position(self, strategy, signal) -> Optional[Action]:
        # We've already checked signal is not same direction as position
        # Zero signal means "CLOSE", oppposite signal means "REVERSE"
        return "REVERSE"

    def process_non_zero_signal_no_position(self, strategy, signal) -> Optional[Action]:
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

    * Zero signals ignored
    """

    def process_zero_signal_position(self, strategy, signal) -> Optional[Action]:
        return None

    def process_non_zero_signal_position(self, strategy, signal) -> Optional[Action]:
        # We've already checked signal is not same direction as position
        # Zero signal means "CLOSE", oppposite signal means "REVERSE"
        return "CLOSE"


class AlwaysOnLockableBinarySignalProcessor(LockableBinarySignalProcessor):
    def __init__(
        self,
        signal_fields: Union[str, tuple[str, str], list[str]] = "signal",
        state_machine: Optional[StateMachine] = None,
    ) -> None:
        super().__init__(signal_fields, state_machine)
        if self.signal_in_field != self.signal_out_field:
            log.exception(f"{self} requires single signal field, not {signal_fields}")
            raise

    def process_non_zero_signal_position(self, strategy, signal):
        return "REVERSE"


class AlwaysOnBinarySignalProcessor(BinarySignalProcessor):
    def __init__(
        self,
        signal_fields: Union[str, tuple[str, str], list[str]] = "signal",
        state_machine: Optional[StateMachine] = None,
    ) -> None:
        super().__init__(signal_fields, state_machine)
        if self.signal_in_field != self.signal_out_field:
            log.exception(f"{self} requires single signal field, not {signal_fields}")
            raise

    def process_non_zero_signal_position(self, strategy, signal) -> Optional[Action]:
        return "REVERSE"


def binary_signal_processor_factory(
    lockable=False, always_on=False
) -> Type[AbstractBaseBinarySignalProcessor]:
    if lockable and always_on:
        return AlwaysOnLockableBinarySignalProcessor
    elif lockable:
        return LockableBinarySignalProcessor
    elif always_on:
        return AlwaysOnBinarySignalProcessor
    else:
        return BinarySignalProcessor
