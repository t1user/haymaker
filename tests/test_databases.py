"""Tests for runtime-owned persistence services."""

from unittest.mock import MagicMock

import pytest

from haymaker import databases
from haymaker.config.settings import MongoSettings, StorageSettings
from haymaker.databases import StoreFactory


def storage_settings(**client: object) -> StorageSettings:
    """Return storage settings with one test Mongo database."""

    return StorageSettings(mongodb=MongoSettings(client=client, database="test_data"))


def test_real_mongo_client_blocked_by_default() -> None:
    factory = StoreFactory(storage_settings())

    with pytest.raises(AssertionError, match="real MongoDB"):
        factory.mongo_client()


def test_store_factory_creates_and_reuses_one_probed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    created_with: dict[str, object] = {}

    def fake_mongo_client(**kwargs: object) -> object:
        created_with.update(kwargs)
        return client

    monkeypatch.setattr(databases, "MongoClient", fake_mongo_client)
    factory = StoreFactory(storage_settings(host="mongodb://example"))

    assert factory.mongo_client() is client
    assert factory.mongo_client() is client

    assert created_with == {"host": "mongodb://example"}
    client.admin.command.assert_called_once_with("ping")
    assert factory.health_checks == [factory.mongodb_health_check]


def test_store_factory_reraises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_mongo_client(**kwargs: object) -> object:
        raise databases.ConfigurationError("bad mongo configuration")

    monkeypatch.setattr(databases, "MongoClient", fake_mongo_client)
    factory = StoreFactory(storage_settings())

    with pytest.raises(databases.ConfigurationError):
        factory.mongo_client()


def test_database_name_is_required_only_for_savers() -> None:
    factory = StoreFactory(StorageSettings())

    with pytest.raises(ValueError, match="database"):
        _ = factory.database
