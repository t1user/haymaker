from ..date import CLOSED_OPEN as CLOSED_OPEN, DateRange as DateRange, OPEN_CLOSED as OPEN_CLOSED, to_pandas_closed_closed as to_pandas_closed_closed
from ..hosts import get_arctic_lib as get_arctic_lib
from .utils import setup_logging as setup_logging
from _typeshed import Incomplete
from arctic.decorators import _get_host as _get_host
from arctic.store.audit import ArcticTransaction as ArcticTransaction

logger: Incomplete
USER: Incomplete

def copy_symbols_helper(src, dest, log, force, splice): ...
def main() -> None: ...
