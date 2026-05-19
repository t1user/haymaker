from unittest.mock import MagicMock

import pytest

from haymaker import databases


def test_get_mongo_client_uses_public_mongo_client_symbol(monkeypatch):
    """Verify that client construction goes through the public MongoClient import."""
    client = MagicMock()
    created_with = {}

    def fake_mongo_client(**kwargs):
        """Capture constructor kwargs and return a mock client."""
        created_with.update(kwargs)
        return client

    monkeypatch.setattr(databases, "MONGODB_CONFIG", {"host": "mongodb://example"})
    monkeypatch.setattr(databases, "MongoClient", fake_mongo_client)
    monkeypatch.setattr(databases, "HEALTH_CHECK_OBSERVABLES", [])
    databases.get_mongo_client.cache_clear()

    try:
        assert databases.get_mongo_client() is client
    finally:
        databases.get_mongo_client.cache_clear()

    assert created_with == {"host": "mongodb://example"}
    client.admin.command.assert_called_once_with("ping")
    assert databases.HEALTH_CHECK_OBSERVABLES == [databases.mongodb_health_check]


def test_get_mongo_client_reraises_configuration_error(monkeypatch):
    """Verify that PyMongo configuration errors are preserved for callers."""

    def fake_mongo_client(**kwargs):
        """Raise the same public PyMongo error type used by the production import."""
        raise databases.ConfigurationError("bad mongo configuration")

    monkeypatch.setattr(databases, "MongoClient", fake_mongo_client)
    databases.get_mongo_client.cache_clear()

    try:
        with pytest.raises(databases.ConfigurationError):
            databases.get_mongo_client()
    finally:
        databases.get_mongo_client.cache_clear()
