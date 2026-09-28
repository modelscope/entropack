# Configuration file for the Sphinx documentation builder.

import tomllib
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "entropack"
copyright = "2026, EntroPack Authors"
author = "EntroPack Authors"
html_theme = "sphinx_rtd_theme"
language = "en"


def get_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


version = get_version()
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_markdown_tables",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "myst_parser",
]

source_suffix = [".rst", ".md"]
root_doc = "index"
exclude_patterns = ["build", "_build"]

# -- Extension configuration -------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
intersphinx_mapping = {"https://docs.python.org/": None}
myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence"]
