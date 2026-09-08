"""Keep documentation type aliases explicit and unknown references visible."""

import pickle
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def formatter(monkeypatch):
    """Load the documentation-only callable as Sphinx does."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "docs"))
    from sphinx_helpers import PublicTypeFormatter

    return PublicTypeFormatter()


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (pd.DataFrame, ":py:class:`pandas.DataFrame`"),
        (pd.Series, ":py:class:`pandas.Series`"),
        (int, None),
    ],
)
def test_public_type_formatter_preserves_unknown_annotations(
    formatter, annotation, expected
):
    """Canonical aliases resolve, but unrelated types retain normal validation."""
    assert formatter(annotation) == expected


def test_formatter_survives_sphinx_configuration_cache(formatter):
    """A cached build uses the same formatter rather than dropping its config."""
    restored = pickle.loads(pickle.dumps(formatter))
    assert restored(pd.DataFrame) == formatter(pd.DataFrame)
