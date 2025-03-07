import os
import sys
import tempfile
from collections import ChainMap
from pathlib import Path

import pytest
import yaml

from haymaker.config.config import ConfigMaps


@pytest.fixture
def temp_yaml_file(tmp_path):
    """Creates a temporary YAML file for testing."""
    data = {"key": "value", "nested": {"subkey": "subvalue"}}
    file_path = tmp_path / "test_config.yaml"
    print(tmp_path)
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def clear_env():
    """Ensure environment variables are cleared before each test."""
    old_env = os.environ.copy()
    os.environ = {k: v for k, v in os.environ.items() if not k.startswith("HAYMAKER_")}
    yield
    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture
def reset_sys_argv():
    """Ensure sys.argv is restored after tests."""
    original_argv = sys.argv.copy()
    yield
    sys.argv = original_argv


def test_env_variable_loading(clear_env):
    # `HAYMAKER_` needs to be chopped off
    # all keys are small caps
    os.environ["HAYMAKER_TEST_KEY"] = "test_value"
    config = ConfigMaps()
    assert config.environ["test_key"] == "test_value"


def test_cmdline_parsing(reset_sys_argv):
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
        temp_path = Path(temp_file.name)  # Get the temporary file path

    try:
        sys.argv = [
            "script_name",
            "--file",
            str(temp_path),
            "--set_option",
            "option",
            "value",
        ]
        config = ConfigMaps()

        assert config.cmdline["file"] == str(temp_path)
        assert config.cmdline["option"] == "value"
    finally:
        temp_path.unlink()  # Cleanup the temporary file


def test_yaml_parsing(temp_yaml_file):
    config = ConfigMaps()
    parsed_yaml = config.parse_yaml(temp_yaml_file)
    assert parsed_yaml["key"] == "value"
    assert parsed_yaml["nested"]["subkey"] == "subvalue"


def test_missing_yaml_file():
    config = ConfigMaps()
    with pytest.raises(FileNotFoundError):
        config.parse_yaml("non_existent.yaml")


def test_config_merging(monkeypatch):
    monkeypatch.setenv("HAYMAKER_TEST_KEY", "env_value")
    sys.argv = ["script_name", "--set_option", "option", "cmdline_value"]
    config_maps = ConfigMaps()
    merged = ChainMap(*config_maps.maps)
    assert merged["option"] == "cmdline_value"
    assert merged.get("test_key") == "env_value"


def test_priorities_1(monkeypatch):
    # cmdline overrides env value
    monkeypatch.setenv("HAYMAKER_TEST_KEY", "env_value")
    sys.argv = ["script_name", "--set_option", "test_key", "cmdline_value"]
    config_maps = ConfigMaps()
    merged = ChainMap(*config_maps.maps)
    assert merged["test_key"] == "cmdline_value"


def test_priorities_2(monkeypatch, temp_yaml_file, reset_sys_argv):
    # config file has priority before env
    monkeypatch.setenv("HAYMAKER_KEY", "env_value")
    # config file name passed as cli argument
    sys.argv = [
        "script_name",
        "--file",
        str(temp_yaml_file),
    ]
    config_maps = ConfigMaps()
    merged = ChainMap(*config_maps.maps)
    assert merged.get("key") == "value"


def test_config_file_read_from_env(monkeypatch, temp_yaml_file, reset_sys_argv):
    # config file is read from env
    monkeypatch.setenv("HAYMAKER_KEY", "env_value")
    monkeypatch.setenv("HAYMAKER_HAYMAKER_CONFIG_OVERRIDES", str(temp_yaml_file))
    config_maps = ConfigMaps()
    merged = ChainMap(*config_maps.maps)
    # it would be 'env_value' if it wasn't overridden with config file
    assert merged.get("key") == "value"
