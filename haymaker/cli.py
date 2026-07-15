from __future__ import annotations

import importlib.util
import sys
from collections.abc import MutableMapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

from .config import configure
from .config.cli_options import CustomArgParser, ParserProfile

if TYPE_CHECKING:
    from .app import Runtime
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
    config: MutableMapping, module_path: str | Path
) -> tuple[Runtime, ConnectionSettings]:
    """Build a live runtime, then import its user strategy module."""

    from .runtime import LiveRuntime
    from .supervisor import ConnectionSettings

    runtime = LiveRuntime(config)
    load_user_module(module_path)
    settings = ConnectionSettings.from_config(config.get("app") or {}, 0)
    return runtime, settings


def _build_dataloader_runtime(
    config: MutableMapping,
) -> tuple[Runtime, ConnectionSettings]:
    """Build the configured standalone dataloader runtime."""

    from .dataloader import dataloader
    from .supervisor import ConnectionSettings

    runtime = dataloader.create_runtime()
    settings = ConnectionSettings.from_config(
        config, config.get("clientId", dataloader.DEFAULT_CLIENT_ID)
    )
    return runtime, settings


def _run_command(
    profile: ParserProfile,
    args: list[str],
    module_path: str | Path | None = None,
) -> None:
    """Configure and run one command profile through the shared application."""

    config = configure(profile, args)

    from .logging import setup_logging, shutdown_logging_queue

    setup_logging(config.get("logging_config"))
    try:
        from .app import App

        if profile == "live":
            if module_path is None:
                raise ValueError("Live execution requires a strategy module path.")
            runtime, settings = _build_live_runtime(config, module_path)
        else:
            runtime, settings = _build_dataloader_runtime(config)
        App(runtime, settings).run()
    finally:
        shutdown_logging_queue()


def main(argv: list[str] | None = None) -> None:
    """Run a live Haymaker strategy module under framework control."""

    args = list(sys.argv[1:] if argv is None else argv)
    parsed = CustomArgParser.from_profile("live", args).output
    _run_command("live", args, parsed["module_path"])


def dataloader_main(argv: list[str] | None = None) -> None:
    """Run the dataloader with explicit dataloader CLI/config profile."""

    args = list(sys.argv[1:] if argv is None else argv)
    _run_command("dataloader", args)
