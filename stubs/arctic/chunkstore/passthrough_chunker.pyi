from ._chunker import Chunker as Chunker
from _typeshed import Incomplete
from collections.abc import Generator

class PassthroughChunker(Chunker):
    TYPE: str
    def to_chunks(self, df, **kwargs) -> Generator[Incomplete, None, None]: ...
    def to_range(self, start, end): ...
    def chunk_to_str(self, chunk_id): ...
    def to_mongo(self, range_obj): ...
    def filter(self, data, range_obj): ...
    def exclude(self, data, range_obj): ...