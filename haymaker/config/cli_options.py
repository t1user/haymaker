from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass, field
from typing import Optional


class SetValues(argparse.Action):
    """
    To be used as `action` on argparser set_option.  Allows to pass
    key, value pairs from commandline.
    """

    def __init__(self, option_strings, dest, nargs=2, **kwargs):
        if nargs != 2:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, values[0], values[1])


class SetSource(argparse.Action):
    """
    To be used as `action` on argparser source.  When no source
    passed, don't put the default None into the resulting dict to give
    other methods a chance to define the value.
    """

    def __call__(self, parser, namespace, values, option_string=None):

        if values is None:
            del namespace[self.dest]
        else:
            setattr(namespace, self.dest, values)


help_string = (
    "Primary way to set up appplication is through a yaml file, or "
    "environment variables. Command line options listed here override"
    "those default settings."
)
epilog = "Defaults will be used for all unset parameters."


common_options = [
    (
        ("source",),
        {
            # "action": SetSource,
            "nargs": "?",
            "help": "Optional file with source data.",
        },
    ),
    (
        ("-s", "--set_option"),
        {
            "action": SetValues,
            "metavar": "ARG",
            "help": "Use provided KEY VALUE pair to set parameter KEY to VALUE.",
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
    "dataloader": [
        (
            ("-g", "--fill_gaps"),
            {
                "action": "store_true",
                "default": False,
                "help": "Whether dataloader should attempt to patch any existing gap in data.",
            },
        ),
        (
            ("-w", "--watchdog"),
            {
                "action": "store_true",
                "default": False,
                "help": "Whether watchdog should be used to monitor ib_gateway.",
            },
        ),
    ],
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
        self.parse()

    @property
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=help_string, epilog=epilog)
        if self.options:
            for option in self.options:
                parser.add_argument(*option[0], **option[1])
        return parser

    def parse(self) -> None:
        if "test" in self.file_name or "sphinx" in self.file_name:
            return
        else:
            self.output = {
                k: v
                for k, v in vars(self.parser.parse_args(self.argv)).items()
                if v is not None
            }

    @classmethod
    def from_args(cls, args: list[str]) -> CustomArgParser:
        """
        Create parser from ``sys.argv``

        Args:
           args: value of sys.argv
        """
        file_name = pathlib.Path(args[0]).name
        if (options := options_by_module.get(file_name.strip(".py"))) is not None:
            return cls(file_name, args[1:], [*common_options, *options])
        else:
            return cls(
                file_name, args[1:], [*common_options, *options_by_module["app"]]
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


# Below is some nonsense to facilitate docs generation


def get_parser_for_dataloader():
    # Simulate running as 'dataloader.py'
    parser_instance = CustomArgParser.from_args(["dataloader.py"])
    return parser_instance.parser


def get_parser_for_other_module():
    # Simulate running as 'your_module.py'
    parser_instance = CustomArgParser.from_args(["your_module.py"])
    return parser_instance.parser
