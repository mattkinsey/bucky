# Configuration file for the Sphinx documentation builder.
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

import recommonmark

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import sphinx_rtd_theme
from recommonmark.transform import AutoStructify

# sys.path.insert(0, os.path.abspath('../../..'))
sys.path.insert(0, os.path.abspath("../.."))
repo_root_url = "https://github.com/OCHA-DAP/pa-ocha-bucky"


repo_root_url = "http://gitlab.com/kinsemc/bucky/"

# -- Project information -----------------------------------------------------

project = "OCHA-Bucky"
copyright = "2020, The Johns Hopkins University Applied Physics Laboratory LLC"
author = "Matt Kinsey, Kate Tallaksen, Freddy Obrecht"

# The full version, including alpha/beta/rc tags
release = ".1"

# change the name of index for RTD
master_doc = "index"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "recommonmark",
    "sphinxcontrib.tikz",
    "sphinxarg.ext",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

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

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = 'alabaster'
html_logo = "../../logo.png"
# html_theme_options = {
#        'logo_only':True,
#    }
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            "url_resolver": lambda url: repo_doc_root + url,
            "auto_toc_tree_section": "Usage",
        },
        True,
    )
    app.add_transform(AutoStructify)
