"""Public documentation contract for the component toolbox."""

import inspect

from haymaker import components
from haymaker.base import Atom, Pipe


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
