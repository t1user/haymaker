from ..arctic import Arctic as Arctic, ArcticLibraryBinding as ArcticLibraryBinding
from ..hooks import get_mongodb_uri as get_mongodb_uri
from .utils import do_db_auth as do_db_auth, setup_logging as setup_logging
from _typeshed import Incomplete

logger: Incomplete

def prune_versions(lib, symbols, keep_mins) -> None: ...
def main() -> None: ...
