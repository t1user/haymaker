from _typeshed import Incomplete
from typing import NamedTuple


class VersionedItem(NamedTuple):
    def metadata_dict(self):
        ...

    def __repr__(self):
        ...

    def __str__(self):
        ...


class ChangedItem(NamedTuple):
    symbol: Incomplete
    orig_version: Incomplete
    new_version: Incomplete
    changes: Incomplete
