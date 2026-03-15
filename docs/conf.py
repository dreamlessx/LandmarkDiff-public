# Sphinx configuration for LandmarkDiff documentation

project = "LandmarkDiff"
author = "dreamlessx"
copyright = "2024-2026, dreamlessx"
release = "0.2.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# MyST settings for Markdown support
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# HTML theme
html_theme = "furo"
html_title = "LandmarkDiff"
html_theme_options = {
    "source_repository": "https://github.com/dreamlessx/LandmarkDiff-public",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
set_type_checking_flag = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Copybutton settings - skip shell prompts
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
