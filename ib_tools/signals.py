from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import Protocol

import numpy as np

from ib_tools.manager import STATE_MACHINE
from ib_tools.misc import PS, P


class SignalProcessor(ABC):
    """
    Any processing that needs to happen between `Brick` and
    `Portfolio` objects.  It's the user's responsibility to make sure
    that signal sent by Brick and processed by SignalProcessor can be
    correctly interpreted by Portfolio.

    If either `Brick` or `Portfolio` needs to check current state or
    market situation, it should be done here.

    It's not obligatory for `Brick` to use :class:`.SignalProcessor`

    Returns:
    ========

    (dict) Dict with keys interpretable by `Portfolio` and `ExecModel`.
    """

    def __call__(self, func):
        """
        The object can be used as a decorator on the signal
        producing method of Brick.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            signal, context = func(*args, **kwargs)
            return self.process_signal(signal, context)

        return wrapper

    @abstractmethod
    def process_signal(self, signal: P, context) -> PS:
        """
        Given signal, should return desired position.
        """
        ...

    def __repr__(self):
        return self.__class__.__name__ + "()"


class BinarySignalContext(Protocol):
    key: str
    lockable: bool
    always_on: bool


class StateCheckerProtocol(Protocol):
    def position(self, key: str) -> P:
        ...

    def locked(self, key: str) -> bool:
        ...


class LockableMixin:
    pass


class AlwaysOnMixin:
    pass


class BinarySignalProcessor(SignalProcessor):
    def __init__(self) -> None:
        self.sm = STATE_MACHINE

    def process_signal(self, signal: P, context: BinarySignalContext) -> PS:
        if self.position(signal, context):
            return self.process_position(signal, context)
        else:
            return self.proces_no_position(signal, context)

    def position(self, signal: P, context: BinarySignalContext) -> P:
        return np.sign(self.sm.position(context.key))

    def locked(self, signal: P, context: BinarySignalContext) -> bool:
        return self.sm.locked(context.key)

    def direction(self, signal: P, context: BinarySignalContext) -> bool:
        return signal == self.position

    def same_direction(self, signal: P, context: BinarySignalContext) -> bool:
        return self.position(signal, context) == signal

    def process_position(self, signal: P, context: BinarySignalContext) -> PS:
        if signal == 0:
            if context.lockable:
                return 0, -self.position(signal, context), "close"
            else:
                return self.position(signal, context), 0, None
        elif self.same_direction(signal, context):
            return signal, 0, None
        elif context.always_on:
            return signal, 2 * signal, "reverse"  # type: ignore
        else:
            return 0, signal, "close"

    def proces_no_position(self, signal: P, context: BinarySignalContext) -> PS:
        if context.lockable & (self.locked(signal, context) == signal):
            return 0, 0, None
        else:
            if signal != 0:
                return signal, signal, "entry"
            else:
                return signal, signal, None
