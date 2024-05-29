from __future__ import annotations

import argparse
import collections.abc
import logging
import os
import sys
from collections import ChainMap
from pathlib import Path
from typing import Final, Optional

import yaml

log = logging.getLogger(__name__)

module_directory = Path(__file__).parents[0]


class Config(ChainMap):
    def __init__(self) -> None:
        self.file: Optional[str] = None
        super().__init__(self.cmdline, self.config_file, self.environ, self.defaults)

    @property
    def cmdline(self) -> collections.abc.MutableMapping:
        if "test" in sys.argv[0]:
            return {}
        argv = sys.argv[1:]
        parser = argparse.ArgumentParser(description="File with config or coldstart.")
        parser.add_argument("-f", "--file", type=str, nargs="?")
        parser.add_argument(
            "-r",
            "--reset",
            action="store_true",
            default=False,
            help="On start, will close all existing positions and cancel orders.",
        )
        parser.add_argument(
            "-z",
            "--zero",
            action="store_true",
            default=False,
            help="On startup will zero all records.",
        )

        parser.add_argument(
            "-c",
            "--coldstart",
            action="store_true",
            default=False,
            help="Start programme without reading state from database.",
        )
        cmdline = parser.parse_args(argv)
        if cmdline.file:
            self.file = Path.cwd() / cmdline.file
        return {
            "coldstart": cmdline.coldstart,
            "reset": cmdline.reset,
            "zero": cmdline.zero,
        }

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

    def parse_keys(self, d: dict) -> collections.abc.Mapping:
        new_dict: dict = {}
        return new_dict


CONFIG: Final[Config] = Config()
