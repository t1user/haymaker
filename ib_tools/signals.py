from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional, Protocol, Tuple

from ib_tools.misc import P, S

PS = Tuple[P, S, Optional[str]]


class SignalProcessor(ABC):
    """
    Any processing that needs to happen between Brick and Portfolio
    objects. It's the user's responsibility to make sure that signal
    sent by Brick and processed by SignalProcessor can be correctly
    interpreted by Portfolio.
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
        return self.__class__.__name__


class BinarySignalContext(Protocol):
    key: tuple[str, str]
    lockable: bool
    always_on: bool


class StateCheckerProtocol(Protocol):
    def position(self, key: tuple[str, str]) -> P:
        ...

    def locked(self, key: tuple[str, str]) -> bool:
        ...


class BinarySignalProcessor(SignalProcessor):
    def __init__(self, state_checker: StateCheckerProtocol):
        self.sm = state_checker

    def process_signal(self, signal: P, context: BinarySignalContext) -> PS:
        print(signal, context)
        if self.position(signal, context):
            return self.process_position(signal, context)
        else:
            return self.proces_no_position(signal, context)

    def position(self, signal: P, context: BinarySignalContext) -> P:
        position = self.sm.position(context.key)
        return position

    def locked(self, signal: P, context: BinarySignalContext) -> bool:
        return self.sm.locked(context.key)

    def direction(self, signal: P, context: BinarySignalContext) -> bool:
        return signal == self.position

    def same_direction(self, signal: P, context: BinarySignalContext) -> bool:
        return self.sm.position(context.key) == signal

    def process_position(self, signal: P, context: BinarySignalContext) -> PS:
        if self.same_direction(signal, context):
            return signal, 0, None
        elif context.always_on:
            return signal, 2 * signal, "reverse"  # type: ignore
        else:
            return 0, signal, "close"

    def proces_no_position(self, signal: P, context: BinarySignalContext) -> PS:
        if context.lockable & (self.locked(signal, context) == signal):
            return 0, 0, None
        else:
            return signal, signal, "entry"
