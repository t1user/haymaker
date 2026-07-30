"""Public documentation contract for the component toolbox."""

import inspect
from types import ModuleType

from haymaker import components
from haymaker.base import Atom, Pipe


def test_component_modules_own_the_package_exports():
    exports = [
        (module, name)
        for module in components._PUBLIC_MODULES
        for name in module.__all__
    ]
    names = [name for _, name in exports]

    assert len(names) == len(set(names))
    assert components.__all__ == names
    assert {
        name
        for name, value in vars(components).items()
        if not name.startswith("_") and not isinstance(value, ModuleType)
    } == set(names)
    for module, name in exports:
        assert getattr(components, name) is getattr(module, name)


def test_every_public_component_export_has_a_docstring():
    missing = [
        name
        for name in components.__all__
        if not inspect.getdoc(getattr(components, name))
    ]

    assert missing == []


def test_atom_and_pipe_have_public_docstrings():
    assert inspect.getdoc(Atom)
    assert inspect.getdoc(Pipe)
