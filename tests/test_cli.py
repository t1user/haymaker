"""Tests for side-effect-free CLI composition."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from haymaker import cli
from haymaker.cli import load_user_module
from haymaker.config import (
    DataloaderCommand,
    LiveCommand,
    load_dataloader_config,
    load_live_config,
)


def test_load_user_module_supports_sibling_imports(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 42\n")
    strategy = tmp_path / "strategy.py"
    strategy.write_text("from helper import VALUE\nresult = VALUE\n")

    module = load_user_module(strategy)

    assert module.result == 42


def test_load_user_module_restores_previous_module_after_failure(
    tmp_path: Path,
) -> None:
    strategy = tmp_path / "strategy.py"
    strategy.write_text("raise RuntimeError('broken strategy')\n")
    module_name = "haymaker_user_strategy"
    previous = ModuleType(module_name)
    sys.modules[module_name] = previous

    try:
        with pytest.raises(RuntimeError, match="broken strategy"):
            load_user_module(strategy)

        assert sys.modules[module_name] is previous
    finally:
        sys.modules.pop(module_name, None)


def test_build_live_runtime_installs_context_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_live_config(LiveCommand(Path("strategy.py"), None, ()), environ={})
    calls: list[tuple[str, object]] = []
    runtime = object()

    def create_runtime(received: object) -> object:
        calls.append(("runtime", received))
        return runtime

    def load(module_path: object) -> ModuleType:
        calls.append(("load", module_path))
        return ModuleType("strategy")

    monkeypatch.setattr(cli, "LiveRuntime", create_runtime)
    monkeypatch.setattr(cli, "load_user_module", load)

    built, connection = cli._build_live_runtime(config, "strategy.py")

    assert built is runtime
    assert connection.client_id == 0
    assert calls == [("runtime", config), ("load", "strategy.py")]


def test_main_runs_merged_config_under_framework_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = LiveCommand(Path("strategy.py"), None, ())
    config = load_live_config(command, environ={})
    runtime = object()
    connection = object()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "parse_live_args", lambda argv: command)

    def load_settings(received: LiveCommand) -> object:
        calls.append(("load_settings", received))
        return config

    def setup_logging(**kwargs: object) -> None:
        calls.append(("setup_logging", kwargs))

    def build(received: object, module_path: object) -> tuple[object, object]:
        calls.append(("build", (received, module_path)))
        return runtime, connection

    class FakeApp:
        def __init__(self, received_runtime: object, connection: object) -> None:
            calls.append(("app", (received_runtime, connection)))

        def run(self) -> None:
            calls.append(("run", None))

    monkeypatch.setattr(cli, "load_live_config", load_settings)
    monkeypatch.setattr(cli, "setup_logging", setup_logging)
    monkeypatch.setattr(cli, "_build_live_runtime", build)
    monkeypatch.setattr(cli, "App", FakeApp)
    monkeypatch.setattr(
        cli,
        "shutdown_logging_queue",
        lambda: calls.append(("shutdown", None)),
    )

    cli.main(["strategy.py"])

    assert calls == [
        ("load_settings", command),
        (
            "setup_logging",
            {
                "config_file": "logging_config.yaml",
                "directory": "logs",
                "level": None,
                "base_directory": config.storage.base_directory,
            },
        ),
        ("build", (config, command.module_path)),
        ("app", (runtime, connection)),
        ("run", None),
        ("shutdown", None),
    ]


def test_main_flushes_logging_when_runtime_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = LiveCommand(Path("strategy.py"), None, ())
    config = load_live_config(command, environ={})
    calls: list[str] = []

    monkeypatch.setattr(cli, "parse_live_args", lambda argv: command)
    monkeypatch.setattr(cli, "load_live_config", lambda command: config)
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: calls.append("setup"))

    def fail(*args: object) -> tuple[Any, Any]:
        calls.append("runtime")
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(cli, "_build_live_runtime", fail)
    monkeypatch.setattr(cli, "shutdown_logging_queue", lambda: calls.append("shutdown"))

    with pytest.raises(RuntimeError, match="runtime failed"):
        cli.main(["strategy.py"])

    assert calls == ["setup", "runtime", "shutdown"]


def test_build_dataloader_runtime_constructs_connection_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_dataloader_config(DataloaderCommand(None, ()), environ={})
    calls: list[object] = []
    runtime = object()

    def create_runtime(received: object) -> object:
        calls.append(received)
        return runtime

    monkeypatch.setattr(cli, "DataloaderRuntime", create_runtime)

    built, connection = cli._build_dataloader_runtime(config)

    assert built is runtime
    assert connection.client_id == 1
    assert connection.app_timeout == 600
    assert calls == [config]


def test_dataloader_main_uses_merged_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = DataloaderCommand(None, ())
    config = load_dataloader_config(command, environ={})
    runtime = object()
    connection = object()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "parse_dataloader_args", lambda argv: command)
    monkeypatch.setattr(cli, "load_dataloader_config", lambda command: config)
    monkeypatch.setattr(
        cli,
        "setup_logging",
        lambda **kwargs: calls.append(("setup", kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "_build_dataloader_runtime",
        lambda received: (runtime, connection),
    )

    class FakeApp:
        def __init__(self, received_runtime: object, connection: object) -> None:
            calls.append(("app", (received_runtime, connection)))

        def run(self) -> None:
            calls.append(("run", None))

    monkeypatch.setattr(cli, "App", FakeApp)
    monkeypatch.setattr(
        cli, "shutdown_logging_queue", lambda: calls.append(("shutdown", None))
    )

    cli.dataloader_main([])

    assert calls == [
        (
            "setup",
            {
                "config_file": "dataloader_logging_config.yaml",
                "directory": "logs",
                "level": None,
                "base_directory": config.storage.base_directory,
            },
        ),
        ("app", (runtime, connection)),
        ("run", None),
        ("shutdown", None),
    ]
