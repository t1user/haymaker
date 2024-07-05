from __future__ import annotations

import collections.abc
import logging
import os
import sys
from collections import ChainMap
from pathlib import Path
from typing import Final, Optional

import yaml

from .cli_options import CustomArgParser

log = logging.getLogger(__name__)

module_directory = Path(__file__).parents[0]


class Config(ChainMap):
    """
    App parameters are determined in the following order:

        - command line

        - yaml file passed from command line

        - yaml file identified by full absolute path in environment
          variable: 'HAYMAKER_CONFIG_OVERRIDES' for modules other than
          dataloader, and `DATALOADER_CONFIG_OVERRIDES` for dataloader

        - environment variables

        - default yaml file (located in the same directory as
          :class:`.Config`) named 'base_config.yaml'
    """

    def __init__(self) -> None:
        self.file: Optional[str] = self.config_file_name_environ_reader()
        super().__init__(self.cmdline, self.config_file, self.environ, self.defaults)

    @staticmethod
    def config_file_name_environ_reader():
        module_name = Path(sys.argv[0]).name.split(".")[0]
        environ_key = {"dataloader": "DATALOADER_CONFIG_OVERRIDES"}.get(
            module_name, "HAYMAKER_CONFIG_OVERRIDES"
        )
        return os.environ.get(environ_key, None)

    @property
    def cmdline(self) -> collections.abc.MutableMapping:
        cmdline = CustomArgParser.from_args(sys.argv).output
        if file := cmdline.get("file"):
            self.file = Path.cwd() / file
        return cmdline

    @property
    def config_file(self) -> collections.abc.MutableMapping:
        if self.file:
            try:
                return self.parse_yaml(self.file)
            except FileNotFoundError:
                print()
                print(f"Missing config file: {self.file}")
                if self.file == os.environ.get("HAYMAKER_CONFIG_OVERRIDES"):
                    print(
                        "file defined in environ variable: `HAYMAKER_CONFIG_OVERRIDES`"
                    )
                else:
                    print("file defined as CLI argument.")
                print()
                raise
        else:
            return {}

    @property
    def environ(self) -> collections.abc.MutableMapping:
        return os.environ

    @property
    def base_config_file(self) -> str:
        root = "base_config.yaml"
        filename = Path(sys.argv[0]).name.strip(".py")
        return f"{filename}_{root}"

    @property
    def defaults(self) -> collections.abc.MutableMapping:
        if (module_directory / self.base_config_file).is_file():
            filename = module_directory / self.base_config_file
        else:
            filename = module_directory / "base_config.yaml"
        return self.parse_yaml(filename)

    def parse_yaml(self, filename) -> collections.abc.MutableMapping:
        with open(filename, "r") as f:
            data = yaml.unsafe_load(f)
        if data is not None:
            return data
        else:
            return {}


CONFIG: Final[Config] = Config()
