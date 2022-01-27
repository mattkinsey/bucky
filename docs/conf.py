"""Configuration file for the Sphinx documentation builder."""
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import six

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("sphinxext"))
from github_link import make_linkcode_resolve  # isort:skip  # pylint: disable=wrong-import-position

# import bucky  # isort:skip  # pylint: disable=wrong-import-position

# -- Project information -----------------------------------------------------

project = "Bucky"
copyright = "2020, The Johns Hopkins University Applied Physics Laboratory LLC"  # pylint: disable=redefined-builtin
author = "Matt Kinsey <matt@mkinsey.com>"

# change the name of index for RTD
master_doc = "index"

# -- General configuration ---------------------------------------------------

extensions = [
    # "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "sphinxcontrib.tikz",
    "sphinxarg.ext",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    # "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    # "sphinx.ext.viewcode",
    "autoapi.extension",
]

autoapi_dirs = ["../bucky"]
autoapi_add_toctree_entry = False
autoapi_member_order = "groupwise"
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    "special-members",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_preprocess_types = True

napoleon_type_aliases = {
    "ndarray": ":class:`numpy.ndarray` or :class:`cupy.ndarray` if using CuPy",
}

# Be picky about warnings
nitpicky = True

# Ignores stuff we can't easily resolve on other project's sphinx manuals
nitpick_ignore = []

with open("nitpick-exceptions", encoding="utf-8") as np_f:
    for line in np_f:
        if line.strip() == "" or line.startswith("#"):
            continue
        dtype, target = line.split(None, 1)
        target = target.strip()
        nitpick_ignore.append((dtype, six.u(target)))

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

# numpydoc_show_class_members=False

linkcode_resolve = make_linkcode_resolve(
    "bucky",
    "https://github.com/mattkinsey/bucky/blob/{revision}/{package}/{path}#L{lineno}",
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
tikz_proc_suite = "GhostScript"
tikz_latex_preamble = "\\newcommand{\\asym}{\\alpha_i}\n \
        \\newcommand{\chr}{\\eta_i}\n \
        \\newcommand{\cfr}{\\phi_i}\n \
        \\newcommand{\htime}{\\rho_i}\n \
        \\tikzstyle{square} = [rectangle, minimum width=1.3cm, minimum height=1cm,text centered, draw=black, fill=gray!10]\n \
        \\tikzstyle{arrow} = [thick, ->]"

bibtex_bibfiles = ["refs.bib"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"
html_logo = "../logo.png"
html_theme_options = {
    "show_toc_level": 4,
    "github_url": "https://github.com/mattkinsey/bucky",
    "external_links": [{"name": "Bucky on GitHub", "url": "https://github.com/mattkinsey/bucky"}],
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
