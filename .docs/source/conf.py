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
version = '0.05'
release = '0.05-dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google and NumPy style docstrings
    'sphinx_autodoc_typehints',  # For type hints support
    'nbsphinx',
    'sphinx_readme'
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

latex_engine = 'xelatex'

latex_elements = {
    'preamble': r'''
    \usepackage{fontspec}
    ''',
}

latex_documents = [
    ('index', 'big-graph-dataset.tex', 'Big Graph Dataset Documentation',
     'Alex O. Davies', 'howto'),  # Change 'manual' to 'howto' or other themes as needed
]

bibtex_bibfiles = ['big-graph-project-reference.bib']

html_logo = "_static/bgd-logo-transparent.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

html_context = {
    'display_github': True,
    'github_user': 'alexodavies',
    'github_repo': 'big-graph-dataset'
}

html_baseurl = 'https://big-graph-dataset.readthedocs.io/en/latest/'
readme_src_files = 'index.rst'
readme_docs_url_type = 'code'