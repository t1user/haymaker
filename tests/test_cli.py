import sys
from types import ModuleType
from typing import Any

import pytest

from haymaker import cli
from haymaker.cli import load_user_module, read_no_future_roll_strategies


def test_load_user_module_supports_sibling_imports(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 42\n")
    strategy = tmp_path / "strategy.py"
    strategy.write_text("from helper import VALUE\nresult = VALUE\n")

    module = load_user_module(strategy)

    assert module.result == 42


def test_read_no_future_roll_strategies_accepts_module_list(tmp_path):
    strategy = tmp_path / "strategy.py"
    strategy.write_text("no_future_roll_strategies = ['one', 'two']\n")

    module = load_user_module(strategy)

    assert read_no_future_roll_strategies(module) == ["one", "two"]


def test_read_no_future_roll_strategies_rejects_wrong_type(tmp_path):
    strategy = tmp_path / "strategy.py"
    strategy.write_text("no_future_roll_strategies = 'one'\n")

    module = load_user_module(strategy)

    with pytest.raises(TypeError):
        read_no_future_roll_strategies(module)


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
    contexts: list[object] = []
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

    class FakeController:
        def set_no_future_roll_strategies(self, strategies: list[str]) -> None:
            calls.append(("roll_exclusions", strategies))

    class FakeRuntimeContext:
        def __init__(self, config: object) -> None:
            contexts.append(self)
            calls.append(("context", config))
            self.no_future_roll_strategies: list[str] = []
            self.controller = FakeController()

    class FakeApp:
        def __init__(self, runtime: object, settings: Any) -> None:
            calls.append(("app", (runtime, settings.client_id)))

        def run(self) -> None:
            calls.append(("run", None))

    def fake_live_runtime(context: object) -> object:
        calls.append(("runtime", context))
        return "live-runtime"

    fake_logging = ModuleType("haymaker.logging")
    setattr(fake_logging, "setup_logging", setup_logging)
    setattr(fake_logging, "shutdown_logging_queue", shutdown_logging_queue)
    fake_runtime = ModuleType("haymaker.runtime")
    setattr(fake_runtime, "RuntimeContext", FakeRuntimeContext)
    fake_app = ModuleType("haymaker.app")
    setattr(fake_app, "App", FakeApp)
    setattr(fake_app, "LiveRuntime", fake_live_runtime)

    monkeypatch.setattr(cli, "configure", configure)
    monkeypatch.setattr(cli, "load_user_module", fake_load_user_module)
    monkeypatch.setitem(sys.modules, "haymaker.logging", fake_logging)
    monkeypatch.setitem(sys.modules, "haymaker.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "haymaker.app", fake_app)

    cli.main(["strategy.py", "-f", "config.yaml"])

    assert calls == [
        ("configure", ("live", ["strategy.py", "-f", "config.yaml"])),
        ("setup_logging", "logging.yaml"),
        ("context", {"logging_config": "logging.yaml", "use_blotter": False}),
        ("load", "strategy.py"),
        ("roll_exclusions", ["one", "two"]),
        ("runtime", contexts[0]),
        ("app", ("live-runtime", 0)),
        ("run", None),
        ("shutdown_logging", None),
    ]
    assert getattr(contexts[0], "no_future_roll_strategies") == ["one", "two"]


def test_main_flushes_logging_when_runtime_construction_fails(monkeypatch) -> None:
    calls = []

    def configure(profile: str, args: list[str]) -> dict[str, object]:
        return {"logging_config": "logging.yaml"}

    def setup_logging(logging_config: object) -> None:
        calls.append("setup")

    def shutdown_logging_queue() -> None:
        calls.append("shutdown")

    class FailingRuntimeContext:
        def __init__(self, config: object) -> None:
            calls.append("context")
            raise RuntimeError("context failed")

    fake_logging = ModuleType("haymaker.logging")
    setattr(fake_logging, "setup_logging", setup_logging)
    setattr(fake_logging, "shutdown_logging_queue", shutdown_logging_queue)
    fake_runtime = ModuleType("haymaker.runtime")
    setattr(fake_runtime, "RuntimeContext", FailingRuntimeContext)

    monkeypatch.setattr(cli, "configure", configure)
    monkeypatch.setitem(sys.modules, "haymaker.logging", fake_logging)
    monkeypatch.setitem(sys.modules, "haymaker.runtime", fake_runtime)

    with pytest.raises(RuntimeError, match="context failed"):
        cli.main(["strategy.py"])

    assert calls == ["setup", "context", "shutdown"]
