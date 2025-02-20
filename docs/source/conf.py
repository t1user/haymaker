# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "Haymaker"
copyright = "2025, t1user"
author = "t1user"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Extracts docstrings
    "sphinx.ext.napoleon",  # Supports Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # Adds links to source code
    "sphinx_autodoc_typehints",  # Generates typehints
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",  # Refer to other projects docs
    "sphinxarg.ext",  # Support for argparse
    "sphinx.ext.autosectionlabel",  # Refer to sections using their title
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "eventkit": ("https://eventkit.readthedocs.io/en/latest", None),
    "ibi": ("https://ib-insync.readthedocs.io/", None),
}

github_url = "https://github.com/t1user/haymaker"
