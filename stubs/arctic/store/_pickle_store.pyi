from .._compression import compress_array as compress_array, decompress as decompress
from .._config import MAX_BSON_ENCODE as MAX_BSON_ENCODE, SKIP_BSON_ENCODE_PICKLE_STORE as SKIP_BSON_ENCODE_PICKLE_STORE
from ..exceptions import UnsupportedPickleStoreVersion as UnsupportedPickleStoreVersion
from ._version_store_utils import checksum as checksum, pickle_compat_load as pickle_compat_load, version_base_or_id as version_base_or_id
from _typeshed import Incomplete

_MAGIC_CHUNKED: str
_MAGIC_CHUNKEDV2: str
_CHUNK_SIZE: Incomplete
_HARD_MAX_BSON_ENCODE: Incomplete
logger: Incomplete

class PickleStore:
    @classmethod
    def initialize_library(cls, *args, **kwargs) -> None: ...
    def get_info(self, _version): ...
    def read(self, mongoose_lib, version, symbol, **kwargs): ...
    @staticmethod
    def read_options(): ...
    def write(self, arctic_lib, version, symbol, item, _previous_version) -> None: ...
