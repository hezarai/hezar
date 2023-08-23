# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Hezar'
copyright = '2023, Hezar AI Team & contributors'
author = 'Hezar AI Team & contributors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
suppress_warnings = ["myst.header"]

html_title = "Hezar Documentation"
html_theme_options = {
    "source_repository": "https://github.com/hezarai/hezar/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-foreground-primary": "#2C3E50",
        "color-inline-code-background": "#F1F1F1",
        "color-background-border": "white",
        "color-toc-background": "#F8F9FB"
    },
    "dark_css_variables": {
        "color-foreground-primary": "#CEDDEB",
        "color-inline-code-background": "#1A1C1E",
        "color-background-primary": "#1A1A1B",
        "color-background-border": "#1A1B1C",
        "color-toc-background": "#1A1C1E"
    }
}

pygments_style = "emacs"
pygments_dark_style = "material"
html_theme = 'furo'
html_static_path = ['_static']

html_logo = "hezar_logo.svg"
html_favicon = "hezar_logo.svg"
