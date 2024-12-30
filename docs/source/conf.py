import inspect
import logging
import os
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Tuple, Union
from urllib.error import URLError
from urllib.request import urlretrieve

import sphinx_autodoc_typehints
from docutils import nodes
from jinja2.defaults import DEFAULT_FILTERS
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.domains.python import PyObject, PyTypedField
from sphinx.environment import BuildEnvironment
from sphinx.ext import autosummary

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))
sys.path.insert(0, os.path.abspath("_ext"))
#sys.path.insert(0, os.path.abspath(__file__+'../../../../src'))
#sys.path.insert(0, os.path.abspath(__file__+'../../../../src/CausalEGM'))
#import CausalBGM
import CausalEGM
# remove PyCharm’s old six module

if "six" in sys.modules:
    print(*sys.path, sep="\n")
    for pypath in list(sys.path):
        if any(p in pypath for p in ["PyCharm", "pycharm"]) and "helpers" in pypath:
            sys.path.remove(pypath)
    del sys.modules["six"]


logger = logging.getLogger(__name__)


# ----------------------------- General configuration -------------------------

needs_sphinx = "1.7"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "recommonmark",
    "sphinx_markdown_tables"
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = 'bysource'

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
todo_include_todos = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CausalBGM'
author = 'Qiao Liu'
copyright = f"{datetime.now():%Y}, {author}"
title = "An AI-powered Bayesian generative modeling approach for causal inference in observational studies"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

html_theme_options = dict(navigation_depth=4, logo_only=True)  # Only show the logo
html_logo = 'logo.png'
