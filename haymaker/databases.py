import logging
from functools import cache

from pymongo import MongoClient  # type: ignore
from pymongo.errors import ConfigurationError  # type: ignore

from .config import CONFIG

MONGODB_CONFIG = CONFIG.get("mongodb") or {}


log = logging.getLogger(__name__)


HEALTH_CHECK_OBSERVABLES = []


@cache
def get_mongo_client() -> MongoClient:
    try:
        client = MongoClient(**MONGODB_CONFIG)
        client.admin.command("ping")
        log.debug(f"Started mongodb with parameters: {MONGODB_CONFIG}")
    except ConfigurationError:
        log.critical(f"Wrong mongodb parameters: {MONGODB_CONFIG}")
        raise
    except Exception as e:
        log.critical(f"Failed to initialize MongoDB client: {e}")
        raise
    HEALTH_CHECK_OBSERVABLES.append(mongodb_health_check)
    return client


def mongodb_health_check() -> bool:
    try:
        get_mongo_client().admin.command("ping")
        return True
    except Exception as e:
        log.critical(f"Broken mongodb connection!!! {e}")
        return False
