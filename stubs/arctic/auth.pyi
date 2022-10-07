from _typeshed import Incomplete
from typing import NamedTuple

logger: Incomplete

def authenticate(db, user, password): ...

class Credential(NamedTuple):
    database: Incomplete
    user: Incomplete
    password: Incomplete

def get_auth(host, app_name, database_name): ...
