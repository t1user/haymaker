from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

from .config import configure
from .config.cli_options import CustomArgParser


def load_user_module(module_path: str | Path) -> ModuleType:
    """Load one user strategy module from a filesystem path.

    Args:
        module_path: Python file containing strategy pipeline construction.

    Returns:
        Imported module object.

    Raises:
        FileNotFoundError: If ``module_path`` does not exist.
        ImportError: If the module cannot be loaded from the provided path.
    """

    path = Path(module_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Strategy module not found: {path}")

    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module_name = f"haymaker_user_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy module from: {path}")

    module = importlib.util.module_from_spec(spec)
    had_previous_module = module_name in sys.modules
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if had_previous_module:
            sys.modules[module_name] = cast(ModuleType, previous_module)
        else:
            sys.modules.pop(module_name, None)
        raise
    return module


def read_no_future_roll_strategies(module: ModuleType) -> list[str]:
    """Read and validate futures-roll exclusions from a strategy module.

    Args:
        module: Imported user strategy module.

    Returns:
        Strategy names excluded from automatic futures rolling.

    Raises:
        TypeError: If the module value is not a list of strings.
    """

    strategies = getattr(module, "no_future_roll_strategies", [])
    if strategies is None:
        return []
    if not isinstance(strategies, list) or not all(
        isinstance(strategy, str) for strategy in strategies
    ):
        raise TypeError(
            "Strategy module variable no_future_roll_strategies must be a list[str]."
        )
    return list(strategies)


def main(argv: list[str] | None = None) -> None:
    """Run a live Haymaker strategy module under framework control."""

    args = list(sys.argv[1:] if argv is None else argv)
    parsed = CustomArgParser.from_profile("live", args).output
    config = configure("live", args)

    from .logging import setup_logging, shutdown_logging_queue

    setup_logging(config.get("logging_config"))
    try:
        from .runtime import RuntimeContext

        context = RuntimeContext(config)
        strategy_module = load_user_module(parsed["module_path"])
        context.no_future_roll_strategies = read_no_future_roll_strategies(
            strategy_module
        )
        context.controller.set_no_future_roll_strategies(
            context.no_future_roll_strategies
        )

        from .app import App, LiveRuntime
        from .supervisor import ConnectionSettings

        runtime = LiveRuntime(context)
        settings = ConnectionSettings.from_config(config.get("app") or {}, 0)
        App(runtime, settings).run()
    finally:
        shutdown_logging_queue()


def dataloader_main(argv: list[str] | None = None) -> None:
    """Run the dataloader with explicit dataloader CLI/config profile."""

    args = list(sys.argv[1:] if argv is None else argv)
    config = configure("dataloader", args)

    from .logging import setup_logging, shutdown_logging_queue

    setup_logging(config.get("logging_config"))
    try:
        from .dataloader import dataloader

        dataloader.start()
    finally:
        shutdown_logging_queue()
