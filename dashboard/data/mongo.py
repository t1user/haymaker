"""MongoDB connection helpers for the dashboard."""

from __future__ import annotations

from functools import cache

import pymongo  # type: ignore

from dashboard.config import DashboardConfig


@cache
def get_client(mongo_uri: str) -> pymongo.MongoClient:
    client = pymongo.MongoClient(mongo_uri)
    client.admin.command("ping")
    return client


def get_database(config: DashboardConfig) -> pymongo.database.Database:
    return get_client(config.mongo_uri)[config.mongo_db]

