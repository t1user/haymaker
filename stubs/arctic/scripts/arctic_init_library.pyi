from ..arctic import Arctic as Arctic, ArcticLibraryBinding as ArcticLibraryBinding, LIBRARY_TYPES as LIBRARY_TYPES, VERSION_STORE as VERSION_STORE
from ..hooks import get_mongodb_uri as get_mongodb_uri
from .utils import do_db_auth as do_db_auth, setup_logging as setup_logging
from _typeshed import Incomplete

logger: Incomplete

def main() -> None: ...
