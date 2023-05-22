from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal

from typing_extensions import Protocol

P = Literal[-1, 0, 1]


class SignalProtocol(Protocol):
    key: Any
    signal: Literal[-1, 1]
    lockable: bool
    always_on: bool


@dataclass
class Signal:
    key: str
    signal: Literal[-1, 1]
    lockable: bool = True
    always_on: bool = False


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
            signal = func(*args, **kwargs)
            return self.process_signal(signal)

        return wrapper

    @abstractmethod
    def process_signal(self, signal: SignalProtocol) -> P:
        """
        Given signal, should return desired position.
        """
        ...

    def __repr__(self):
        return self.__class__.__name__


class BinarySignalProcessor(SignalProcessor):
    def __init__(self, state_machine):
        self.sm = state_machine

    def process_signal(self, signal: SignalProtocol) -> P:
        if self.position(signal):
            return self.process_position(signal)
        else:
            return self.proces_no_position(signal)

    def position(self, signal: SignalProtocol) -> P:
        position = self.sm.position(signal.key)
        return position

    def locked(self, signal: SignalProtocol) -> bool:
        return self.sm.locked(signal.key)

    def direction(self, signal: SignalProtocol) -> bool:
        return signal.signal == self.position

    def lockable(self, signal: SignalProtocol) -> bool:
        return signal.lockable

    def same_direction(self, signal: SignalProtocol) -> bool:
        return self.sm.position(signal.key) == signal.signal

    def process_position(self, signal: SignalProtocol) -> P:
        if self.same_direction(signal):
            return signal.signal
        elif signal.always_on:
            return signal.signal
        else:
            return 0

    def proces_no_position(self, signal: SignalProtocol) -> P:
        if self.lockable(signal) & (self.locked(signal) == signal.signal):
            return 0
        else:
            return signal.signal
