import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust this path to point to your project root

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'big-graph-dataset'
copyright = '2024, Alex O. Davies'
author = 'Alex O. Davies'
release = '0.01'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google and NumPy style docstrings
    'sphinx_autodoc_typehints',  # For type hints support
]

# Optional: Configure autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

# Optional: Napoleon settings if using Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
autodoc_inherit_docstrings = False

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
