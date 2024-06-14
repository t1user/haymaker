from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass, field
from typing import Optional


class SetValues(argparse.Action):
    def __init__(self, option_strings, dest, nargs=2, **kwargs):
        if nargs != 2:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, values[0], values[1])


help_string = "Write manual on how to use options."
epilog = ""


common_options = [
    (
        ("-s", "--set_option"),
        {
            "action": SetValues,
            "metavar": "KEY VALUE",
            "help": "Set parameter KEY to VALUE.",
        },
    ),
    (
        ("-f", "--file"),
        {
            "type": str,
            "help": "Name of .yaml file with settings "
            "(which must exist in the current directory).",
        },
    ),
]

options_by_module: dict[str, list] = {
    "app": [
        (
            ("-r", "--reset"),
            {
                "action": "store_true",
                "default": False,
                "help": "On startup close all existing positions and cancel orders.",
            },
        ),
        (
            ("-z", "--zero"),
            {
                "action": "store_true",
                "default": False,
                "help": "On startup zero-out all records.",
            },
        ),
        (
            ("-c", "--coldstart"),
            {
                "action": "store_true",
                "default": False,
                "help": "Start programme without reading state from database.",
            },
        ),
        (
            ("-n", "--nuke"),
            {
                "action": "store_true",
                "default": False,
                "help": "Cancel all orders and close positions then stop trading.",
            },
        ),
    ],
    "dataloader": [],
    # this is for tests, but don't use 'test' in module name
    "my_module": [
        (("-t", "--test_option"), {"action": "store_true", "default": False})
    ],
}


@dataclass
class CustomArgParser:
    file_name: str
    argv: list[str]
    options: list = field(default_factory=list)
    output: dict = field(default_factory=dict)

    def __post_init__(self):
        self.parser = argparse.ArgumentParser(description=help_string, epilog=epilog)
        if self.options:
            for option in self.options:
                self.parser.add_argument(*option[0], **option[1])
        self.parse()

    def parse(self) -> None:
        if "test" not in self.file_name:
            self.output = vars(self.parser.parse_args(self.argv))

    @classmethod
    def from_args(cls, args: list[str]) -> CustomArgParser:
        file_name = pathlib.Path(args[0]).name
        if (options := options_by_module.get(file_name.strip(".py"))) is not None:
            return cls(file_name, args[1:], [*options, *common_options])
        else:
            return cls(
                file_name, args[1:], [*options_by_module["app"], *common_options]
            )

    @classmethod
    def from_str(
        cls, test_string: str, module_name: Optional[str] = None
    ) -> CustomArgParser:
        """This is for testing."""

        test_list = test_string.split()
        if module_name:
            test_list.insert(0, module_name)
        else:
            test_list.insert(0, "")
        return cls.from_args(test_list)
