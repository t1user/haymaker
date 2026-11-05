"""Typed accounting and ordered persistence, coordinated by :class:`Book`.

Use Book for coordinated mutations. Recovery record types have explicit codecs;
collection implementations and the shared writer live in their owning modules.
"""

from .core import Book
from .orders import FillRecord, OrderInfo
from .positions import ContractPosition, SignalPosition, SignalTarget, ProtectionState
from .targets import ContractTarget
from .rolls import FutureRollStage, RollParticipant, RollState
from .persistence import DEFAULT_ORDER_COLLECTION_NAME, DEFAULT_STATE_COLLECTION_NAME

__all__ = [
    "Book",
    "FillRecord",
    "OrderInfo",
    "SignalPosition",
    "SignalTarget",
    "ProtectionState",
    "ContractPosition",
    "ContractTarget",
    "FutureRollStage",
    "RollParticipant",
    "RollState",
    "DEFAULT_ORDER_COLLECTION_NAME",
    "DEFAULT_STATE_COLLECTION_NAME",
]
