from ..auth import authenticate as authenticate, get_auth as get_auth
from _typeshed import Incomplete

logger: Incomplete

def do_db_auth(host, connection, db_name): ...
def setup_logging() -> None: ...
