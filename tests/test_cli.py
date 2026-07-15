import sys
from types import ModuleType
from typing import Any
from unittest.mock import ANY

import pytest

from haymaker import cli
from haymaker.cli import load_user_module
from haymaker.supervisor import ConnectionSettings


def test_load_user_module_supports_sibling_imports(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 42\n")
    strategy = tmp_path / "strategy.py"
    strategy.write_text("from helper import VALUE\nresult = VALUE\n")

    module = load_user_module(strategy)

    assert module.result == 42


def test_load_user_module_restores_previous_module_after_failure(tmp_path):
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


def test_main_runs_strategy_module_under_framework_control(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    runtimes: list[object] = []
    strategy_module = ModuleType("strategy")
    setattr(strategy_module, "no_future_roll_strategies", ["one", "two"])

    def configure(profile: str, args: list[str]) -> dict[str, object]:
        calls.append(("configure", (profile, args)))
        return {"logging_config": "logging.yaml", "use_blotter": False}

    def setup_logging(logging_config: object) -> None:
        calls.append(("setup_logging", logging_config))

    def shutdown_logging_queue() -> None:
        calls.append(("shutdown_logging", None))

    def fake_load_user_module(module_path: object) -> ModuleType:
        calls.append(("load", module_path))
        return strategy_module

    class FakeLiveRuntime:
        def __init__(self, config: object) -> None:
            runtimes.append(self)
            calls.append(("runtime", config))

    class FakeApp:
        def __init__(self, runtime: object, settings: Any) -> None:
            calls.append(("app", (runtime, settings.client_id)))

        def run(self) -> None:
            calls.append(("run", None))

    fake_logging = ModuleType("haymaker.logging")
    setattr(fake_logging, "setup_logging", setup_logging)
    setattr(fake_logging, "shutdown_logging_queue", shutdown_logging_queue)
    fake_runtime = ModuleType("haymaker.runtime")
    setattr(fake_runtime, "LiveRuntime", FakeLiveRuntime)
    fake_app = ModuleType("haymaker.app")
    setattr(fake_app, "App", FakeApp)

    monkeypatch.setattr(cli, "configure", configure)
    monkeypatch.setattr(cli, "load_user_module", fake_load_user_module)
    monkeypatch.setitem(sys.modules, "haymaker.logging", fake_logging)
    monkeypatch.setitem(sys.modules, "haymaker.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "haymaker.app", fake_app)

    cli.main(["strategy.py", "-f", "config.yaml"])

    assert calls == [
        ("configure", ("live", ["strategy.py", "-f", "config.yaml"])),
        ("setup_logging", "logging.yaml"),
        ("runtime", {"logging_config": "logging.yaml", "use_blotter": False}),
        ("load", "strategy.py"),
        ("app", (ANY, 0)),
        ("run", None),
        ("shutdown_logging", None),
    ]
    assert len(runtimes) == 1
    assert calls[4] == ("app", (runtimes[0], 0))


def test_main_flushes_logging_when_runtime_construction_fails(monkeypatch) -> None:
    calls = []

    def configure(profile: str, args: list[str]) -> dict[str, object]:
        return {"logging_config": "logging.yaml"}

    def setup_logging(logging_config: object) -> None:
        calls.append("setup")

    def shutdown_logging_queue() -> None:
        calls.append("shutdown")

    class FailingLiveRuntime:
        def __init__(self, config: object) -> None:
            calls.append("runtime")
            raise RuntimeError("runtime failed")

    fake_logging = ModuleType("haymaker.logging")
    setattr(fake_logging, "setup_logging", setup_logging)
    setattr(fake_logging, "shutdown_logging_queue", shutdown_logging_queue)
    fake_runtime = ModuleType("haymaker.runtime")
    setattr(fake_runtime, "LiveRuntime", FailingLiveRuntime)
    fake_app = ModuleType("haymaker.app")
    setattr(fake_app, "App", object)

    monkeypatch.setattr(cli, "configure", configure)
    monkeypatch.setitem(sys.modules, "haymaker.logging", fake_logging)
    monkeypatch.setitem(sys.modules, "haymaker.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "haymaker.app", fake_app)

    with pytest.raises(RuntimeError, match="runtime failed"):
        cli.main(["strategy.py"])

    assert calls == ["setup", "runtime", "shutdown"]


def test_dataloader_main_uses_shared_application_shell(monkeypatch) -> None:
    """Dataloader command should share configuration, App, and logging flow."""

    calls: list[tuple[str, object]] = []
    runtime = object()

    def configure(profile: str, args: list[str]) -> dict[str, object]:
        calls.append(("configure", (profile, args)))
        return {"logging_config": "dataloader-logging.yaml"}

    def setup_logging(logging_config: object) -> None:
        calls.append(("setup_logging", logging_config))

    def shutdown_logging_queue() -> None:
        calls.append(("shutdown_logging", None))

    def build_runtime(config: object) -> tuple[object, ConnectionSettings]:
        calls.append(("build", config))
        return runtime, ConnectionSettings(client_id=1)

    class FakeApp:
        def __init__(self, received_runtime: object, settings: Any) -> None:
            calls.append(("app", (received_runtime, settings.client_id)))

        def run(self) -> None:
            calls.append(("run", None))

    fake_logging = ModuleType("haymaker.logging")
    setattr(fake_logging, "setup_logging", setup_logging)
    setattr(fake_logging, "shutdown_logging_queue", shutdown_logging_queue)
    fake_app = ModuleType("haymaker.app")
    setattr(fake_app, "App", FakeApp)

    monkeypatch.setattr(cli, "configure", configure)
    monkeypatch.setattr(cli, "_build_dataloader_runtime", build_runtime)
    monkeypatch.setitem(sys.modules, "haymaker.logging", fake_logging)
    monkeypatch.setitem(sys.modules, "haymaker.app", fake_app)

    cli.dataloader_main(["contracts.csv"])

    assert calls == [
        ("configure", ("dataloader", ["contracts.csv"])),
        ("setup_logging", "dataloader-logging.yaml"),
        ("build", {"logging_config": "dataloader-logging.yaml"}),
        ("app", (runtime, 1)),
        ("run", None),
        ("shutdown_logging", None),
    ]
