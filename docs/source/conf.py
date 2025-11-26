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

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "sc-heimdall"
copyright = "Ma Lab @ CMU, 2025"
author = "Shahul Alam"

# The full version, including alpha/beta/rc tags
release = "4.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    # "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_remove_toctrees",
    "sphinx_design",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]


# filetypes to use for documentation
source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
# html_theme = "sphinxawesome_theme" # NOTE: to use this, need to improve night theme + remove copy button

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for included extensions -----------------------------------------

autosummary_generate = True
# nbsphinx_execute = 'never'
# nbsphinx_allow_directives = True
nb_execution_mode = "off"
autodoc_typehints = "description"
remove_from_toctrees = ["api/*"]

autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}


# -- Executable Book Theme options
html_theme_options = {
    "show_navbar_depth": 1,
    "repository_url": "https://github.com/ma-compbio/Heimdall",
    "use_repository_button": True,
}

html_title = "sc-heimdall"

# -- Options for intersphinx _-------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}
