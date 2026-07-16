"""Command-line parsing for Haymaker entrypoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from .loader import ConfigError


class SetOption(argparse.Action):
    """Collect a typed dotted-path framework setting override."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Append one parsed path/value pair to the namespace."""

        del option_string
        path, raw_value = cast(Sequence[str], values)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise argparse.ArgumentError(self, f"invalid YAML value: {exc}") from exc
        overrides = list(getattr(namespace, self.dest) or [])
        overrides.append((path, value))
        setattr(namespace, self.dest, overrides)


@dataclass(frozen=True)
class LiveCommand:
    """CLI inputs required to load and run one live strategy."""

    module_path: Path
    config_file: Path | None
    overrides: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class DataloaderCommand:
    """CLI inputs required to configure one dataloader run."""

    config_file: Path | None
    overrides: tuple[tuple[str, Any], ...]


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add configuration-file and dotted-override options."""

    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="YAML file whose framework settings override bundled defaults.",
    )
    parser.add_argument(
        "-s",
        "--set-option",
        action=SetOption,
        nargs=2,
        default=[],
        metavar=("PATH", "VALUE"),
        help="Override one dotted setting path with a YAML scalar or collection.",
    )


def live_parser() -> argparse.ArgumentParser:
    """Return the parser used by the live entrypoint."""

    parser = argparse.ArgumentParser(
        description="Run a Haymaker strategy under connection supervision."
    )
    parser.add_argument("module_path", type=Path, help="Python strategy module.")
    _add_common_options(parser)
    parser.add_argument("--reset", action="store_true", default=None)
    parser.add_argument("--zero", action="store_true", default=None)
    parser.add_argument("--cold-start", action="store_true", default=None)
    parser.add_argument("--nuke", action="store_true", default=None)
    return parser


def dataloader_parser() -> argparse.ArgumentParser:
    """Return the parser used by the dataloader entrypoint."""

    parser = argparse.ArgumentParser(
        description="Download historical IB data under connection supervision."
    )
    parser.add_argument("source", nargs="?", help="CSV file containing contracts.")
    _add_common_options(parser)
    parser.add_argument(
        "-g",
        "--gap-fill-mode",
        choices=("off", "heuristic", "schedule", "auto"),
        default=None,
    )
    return parser


def _config_file(value: Path | None) -> Path | None:
    """Return an expanded command-line configuration path."""

    return value.expanduser() if value is not None else None


def parse_live_args(argv: list[str]) -> LiveCommand:
    """Parse live CLI arguments into a settings-loader command."""

    namespace = live_parser().parse_args(argv)
    overrides = list(namespace.set_option)
    for option, path in (
        (namespace.reset, "startup.reset"),
        (namespace.zero, "startup.zero"),
        (namespace.cold_start, "startup.cold_start"),
        (namespace.nuke, "startup.nuke"),
    ):
        if option is not None:
            overrides.append((path, option))
    return LiveCommand(
        module_path=namespace.module_path.expanduser(),
        config_file=_config_file(namespace.file),
        overrides=tuple(overrides),
    )


def parse_dataloader_args(argv: list[str]) -> DataloaderCommand:
    """Parse dataloader CLI arguments into a settings-loader command."""

    namespace = dataloader_parser().parse_args(argv)
    overrides = list(namespace.set_option)
    if namespace.source is not None:
        overrides.append(("download.source", namespace.source))
    if namespace.gap_fill_mode is not None:
        overrides.append(("download.gap_fill_mode", namespace.gap_fill_mode))
    return DataloaderCommand(
        config_file=_config_file(namespace.file), overrides=tuple(overrides)
    )


def get_parser_for_dataloader() -> argparse.ArgumentParser:
    """Return the dataloader parser for Sphinx documentation."""

    return dataloader_parser()


def get_parser_for_other_module() -> argparse.ArgumentParser:
    """Return the live parser for Sphinx documentation."""

    return live_parser()


__all__ = [
    "ConfigError",
    "DataloaderCommand",
    "LiveCommand",
    "get_parser_for_dataloader",
    "get_parser_for_other_module",
    "parse_dataloader_args",
    "parse_live_args",
]
