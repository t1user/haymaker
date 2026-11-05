from enum import Enum, StrEnum, auto


class ExecutionMode(StrEnum):
    """Identify the scope of positions managed by a process's execution models.

    SIGNAL manages independently attributed signal positions. CONTRACT manages
    aggregate positions per concrete contract.
    """

    SIGNAL = "SIGNAL"
    CONTRACT = "CONTRACT"


class ActiveNext(Enum):
    ACTIVE = auto()
    NEXT = auto()
    PREVIOUS = auto()

    def __str__(self) -> str:
        return self.name
