from __future__ import annotations

import argparse
import collections.abc
import json
import logging
import os
import sys
from collections import ChainMap
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

module_directory = Path(__file__).parents[0]


class Config(ChainMap):
    def __init__(self) -> None:
        self.file: Optional[str] = None
        super().__init__(self.cmdline, self.config_file, self.environ, self.defaults)

    @property
    def cmdline(self) -> collections.abc.MutableMapping:
        argv = sys.argv[1:]
        parser = argparse.ArgumentParser(description="File with config or coldstart.")
        parser.add_argument("-f", "--file", type=str, nargs="?")
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
        return {"coldstart": cmdline.coldstart}

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


def get_options(argv: list[str] = sys.argv[1:]) -> ChainMap:
    """Four Sources: comand line, file, OS environ, defaults."""
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-c", "--configuration", type=open, nargs="?")
    parser.add_argument("-p", "--playerclass", type=str, nargs="?", default="Simple")
    cmdline = parser.parse_args(argv)

    if cmdline.configuration:
        config_file = json.load(cmdline.configuration)
        cmdline.configuration.close()
    else:
        config_file = {}

    default_path = Path.cwd() / "Chapter_7" / "ch07_defaults.json"
    with default_path.open() as default_file:
        defaults = json.load(default_file)

    combined = ChainMap(vars(cmdline), config_file, os.environ, defaults)
    return combined
