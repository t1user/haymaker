from enum import Enum, auto


class ActiveNext(Enum):
    ACTIVE = auto()
    NEXT = auto()
    PREVIOUS = auto()

    def __str__(self) -> str:
        return self.name
