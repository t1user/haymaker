"""
App parameters are determined in the following order:

    - command line

    - yaml file passed from command line

    - yaml file identified by full absolute path in environment
      variable: 'HAYMAKER_HAYMAKER_CONFIG_OVERRIDES' for live execution
      and `HAYMAKER_DATALOADER_CONFIG_OVERRIDES` for dataloader

    - environment variables

    - default yaml file (located in the same directory as
      :class:`.Config`) named 'base_config.yaml' for live execution
      and 'dataloader_base_config.yaml' for dataloader
"""

from __future__ import annotations

import collections.abc
import logging
import os
import sys
from collections import ChainMap
from pathlib import Path
from typing import Final

import yaml

from .cli_options import CustomArgParser, ParserProfile

log = logging.getLogger(__name__)

module_directory = Path(__file__).parents[0]


class ConfigMaps:

    __slots__ = ["_file", "profile", "cmdline", "config_file", "environ", "defaults"]

    _file: str | None
    profile: ParserProfile
    cmdline: collections.abc.MutableMapping
    config_file: collections.abc.MutableMapping
    environ: collections.abc.MutableMapping
    defaults: collections.abc.MutableMapping

    def __init__(
        self, profile: ParserProfile | None = None, args: list[str] | None = None
    ) -> None:
        self.profile = profile or self.profile_from_program_name(sys.argv[0])
        _environ = self.get_environ()
        self._file = self.config_file_name_environ_reader(_environ, self.profile)

        # keep the priority order here
        self.cmdline = self.get_cmdline(args)
        self.config_file = self.get_config_file()
        self.environ = _environ
        self.defaults = self.get_defaults()

    @property
    def maps(self) -> list[collections.abc.MutableMapping]:
        return [self.cmdline, self.config_file, self.environ, self.defaults]

    @staticmethod
    def profile_from_program_name(program_name: str) -> ParserProfile:
        """Return default config profile for legacy direct module execution."""

        module_name = Path(program_name).name.split(".")[0]
        if module_name == "dataloader":
            return "dataloader"
        return "live"

    @staticmethod
    def config_file_name_environ_reader(
        environ: collections.abc.MutableMapping, profile: ParserProfile
    ) -> str | None:
        """
        Retrieve name of config yaml file from environment.  The name
        of the environment variable with config file name depends on the
        explicit command profile.

        Returns:
            Filename if it's stored in environment variable, None otherwise.
        """
        environ_key = {
            "dataloader": "dataloader_config_overrides",
            "live": "haymaker_config_overrides",
        }[profile]
        return environ.get(environ_key, None)

    def get_cmdline(self, args=None) -> collections.abc.MutableMapping:
        """
        Read and return config options passed as command line
        arguments.  If `file` passed through commandline, set `_file`
        property.

        Arguments:
        ----------

        args - allows passsing arguments in tests without mocking
        `sys.argv`
        """
        if args is None:
            argv = sys.argv[1:]
            # Crucial exceptions to let external tools import Haymaker modules.
            if "test" in sys.argv[0] or "sphinx" in sys.argv[0]:
                return {}
        else:
            argv = args
        cmdline = CustomArgParser.from_profile(self.profile, argv).output
        cmdline.pop("module_path", None)
        if file := cmdline.get("file"):
            self._file = Path.cwd() / file
        return cmdline

    def get_config_file(self) -> collections.abc.MutableMapping:
        """
        Return contents of configuration file, whose name is specified
        in environment variable.
        """
        if self._file:
            try:
                return self.parse_yaml(self._file)
            except FileNotFoundError:
                msg = f"\nMissing config file: {self._file}\n"
                if self._file == os.environ.get("HAYMAKER_HAYMAKER_CONFIG_OVERRIDES"):
                    msg += (
                        "file defined in environ variable: "
                        "`HAYMAKER_HAYMAKER_CONFIG_OVERRIDES`\n"
                    )
                else:
                    msg += "file defined as CLI argument.\n"
                print(f"Error: {msg}")
                sys.stderr.write(msg)
                log.error(msg.strip())
                raise
        else:
            return {}

    def get_environ(self) -> collections.abc.MutableMapping:
        """Return contents of environment."""
        return {
            key[len("HAYMAKER_") :].lower(): value
            for key, value in os.environ.items()
            if key.startswith("HAYMAKER_")
        }

    @property
    def base_config_file(self) -> str:
        root = "base_config.yaml"
        if self.profile == "dataloader":
            return f"dataloader_{root}"
        return root

    def get_defaults(self) -> collections.abc.MutableMapping:
        """
        Return contents of default config file, which will be used to
        set all parameters that have not been defined otherwise.
        """
        if (module_directory / self.base_config_file).is_file():
            filename = module_directory / self.base_config_file
        else:
            filename = module_directory / "base_config.yaml"
        return self.parse_yaml(filename)

    def parse_yaml(self, filename) -> collections.abc.MutableMapping:
        """
        Yaml format reader.
        """
        with open(filename, "r") as f:
            data = yaml.unsafe_load(f)
        return data or {}


CONFIG: Final[ChainMap] = ChainMap(*ConfigMaps().maps)


def configure(
    profile: ParserProfile, args: list[str] | None = None
) -> collections.abc.MutableMapping:
    """Install command-profile configuration into the shared config object.

    Args:
        profile: Haymaker command profile to configure.
        args: Command-line arguments without the executable name.

    Returns:
        The shared ``CONFIG`` mapping after its underlying maps are replaced.
    """

    CONFIG.maps[:] = ConfigMaps(profile, args).maps
    return CONFIG
