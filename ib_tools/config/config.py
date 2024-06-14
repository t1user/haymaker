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

        - environment variables

        - default yaml file (located in the same directory as
          :class:`.Config`)
    """

    def __init__(self) -> None:
        self.file: Optional[str] = None
        super().__init__(self.cmdline, self.config_file, self.environ, self.defaults)

    @property
    def cmdline(self) -> collections.abc.MutableMapping:
        cmdline = CustomArgParser.from_args(sys.argv).output
        if file := cmdline.get("file"):
            self.file = Path.cwd() / file
        return cmdline

    @property
    def config_file(self) -> collections.abc.MutableMapping:
        if self.file:
            return self.parse_yaml(self.file)
        else:
            return {}

    @property
    def environ(self) -> collections.abc.MutableMapping:
        return os.environ

    @property
    def defaults(self) -> collections.abc.MutableMapping:
        filename = module_directory / "base_config.yaml"
        return self.parse_yaml(filename)

    def parse_yaml(self, filename) -> collections.abc.MutableMapping:
        with open(filename, "r") as f:
            data = yaml.unsafe_load(f)
        return data

    # def parse_keys(self, d: dict) -> collections.abc.Mapping:
    #     new_dict: dict = {}
    #     return new_dict


CONFIG: Final[Config] = Config()
