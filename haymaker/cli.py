"""Haymaker live and dataloader command entrypoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

from .app import App, Runtime
from .config import (
    DataloaderCommand,
    DataloaderConfig,
    LiveConfig,
    LiveCommand,
    load_dataloader_config,
    load_live_config,
    parse_dataloader_args,
    parse_live_args,
)
from .dataloader.runtime import DataloaderRuntime
from .logging import setup_logging, shutdown_logging_queue
from .runtime import LiveRuntime
from .supervisor import ConnectionSettings


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


def _build_live_runtime(
    config: LiveConfig, module_path: str | Path
) -> tuple[Runtime, ConnectionSettings]:
    """Build a complete live runtime, then import its strategy module."""

    connection = ConnectionSettings.from_mapping(config.connection)
    runtime = LiveRuntime(config)
    load_user_module(module_path)
    return runtime, connection


def _build_dataloader_runtime(
    config: DataloaderConfig,
) -> tuple[Runtime, ConnectionSettings]:
    """Build the configured standalone dataloader runtime."""

    connection = ConnectionSettings.from_mapping(config.connection)
    return DataloaderRuntime(config), connection


def _run_live(command: LiveCommand, config: LiveConfig) -> None:
    """Run one merged live configuration through the shared application."""

    logging_options = dict(config.logging)
    logging_options.pop("log_broker", None)
    setup_logging(
        **logging_options,
        base_directory=config.storage.base_directory,
    )
    try:
        runtime, connection = _build_live_runtime(config, command.module_path)
        App(runtime, connection).run()
    finally:
        shutdown_logging_queue()


def _run_dataloader(command: DataloaderCommand, config: DataloaderConfig) -> None:
    """Run one merged dataloader configuration through the shared application."""

    del command
    setup_logging(
        **dict(config.logging),
        base_directory=config.storage.base_directory,
    )
    try:
        runtime, connection = _build_dataloader_runtime(config)
        App(runtime, connection).run()
    finally:
        shutdown_logging_queue()


def main(argv: list[str] | None = None) -> None:
    """Run a live Haymaker strategy module under framework control."""

    command = parse_live_args(list(sys.argv[1:] if argv is None else argv))
    _run_live(command, load_live_config(command))


def dataloader_main(argv: list[str] | None = None) -> None:
    """Run the dataloader with explicit dataloader configuration."""

    command = parse_dataloader_args(list(sys.argv[1:] if argv is None else argv))
    _run_dataloader(command, load_dataloader_config(command))
