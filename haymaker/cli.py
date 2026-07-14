from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import ib_insync as ibi

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
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> None:
    """Run a live Haymaker strategy module under framework control."""

    args = list(sys.argv[1:] if argv is None else argv)
    parsed = CustomArgParser.from_profile("live", args).output
    config = configure("live", args)

    from .logging import setup_logging
    from .runtime import RuntimeContext

    setup_logging(config.get("logging_config"))
    ibi.util.patchAsyncio()
    context = RuntimeContext(config)
    strategy_module = load_user_module(parsed["module_path"])
    context.bind_strategy_module(strategy_module)

    from .app import App, LiveRuntime
    from .supervisor import ConnectionSettings

    runtime = LiveRuntime.from_context(context)
    settings = ConnectionSettings.from_config(config.get("app") or {}, 0)
    App(runtime, settings).run()


def dataloader_main(argv: list[str] | None = None) -> None:
    """Run the dataloader with explicit dataloader CLI/config profile."""

    args = list(sys.argv[1:] if argv is None else argv)
    configure("dataloader", args)

    from .dataloader import dataloader

    dataloader.start()
