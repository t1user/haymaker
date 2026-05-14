"""Configuration loading for the local dashboard."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DASHBOARD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_ROOT.parent


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_dashboard_env() -> None:
    """Load local env files without overriding already exported variables."""

    _load_env_file(REPO_ROOT / ".env")
    _load_env_file(DASHBOARD_ROOT / ".env")


@dataclass(frozen=True)
class DashboardConfig:
    mongo_uri: str
    mongo_db: str
    blotter_collection: str
    orders_collection: str
    models_collection: str
    ib_host: str
    ib_port: int
    ib_client_id: int
    ib_account: str


def get_config() -> DashboardConfig:
    load_dashboard_env()
    return DashboardConfig(
        mongo_uri=os.getenv("DASHBOARD_MONGO_URI", "mongodb://localhost:27017"),
        mongo_db=os.getenv("DASHBOARD_MONGO_DB", "walk_forward"),
        blotter_collection=os.getenv("DASHBOARD_BLOTTER_COLLECTION", "blotter"),
        orders_collection=os.getenv("DASHBOARD_ORDERS_COLLECTION", "orders"),
        models_collection=os.getenv("DASHBOARD_MODELS_COLLECTION", "models"),
        ib_host=os.getenv("DASHBOARD_IB_HOST", "localhost"),
        ib_port=int(os.getenv("DASHBOARD_IB_PORT", "4002")),
        ib_client_id=int(os.getenv("DASHBOARD_IB_CLIENT_ID", "102")),
        ib_account=os.getenv("DASHBOARD_IB_ACCOUNT", "DU3598515"),
    )

