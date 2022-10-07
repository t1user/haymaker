from .._compression import compress as compress, compress_array as compress_array, decompress as decompress
from ._serializer import Serializer as Serializer
from _typeshed import Incomplete

DATA: str
MASK: str
TYPE: str
DTYPE: str
COLUMNS: str
INDEX: str
METADATA: str
LENGTHS: str

class FrameConverter:
    def _convert_types(self, a): ...
    def docify(self, df): ...
    def objify(self, doc, columns: Incomplete | None = ...): ...

class FrametoArraySerializer(Serializer):
    TYPE: str
    converter: Incomplete
    def __init__(self) -> None: ...
    def serialize(self, df): ...
    def deserialize(self, data, columns: Incomplete | None = ...): ...
    def combine(self, a, b): ...
