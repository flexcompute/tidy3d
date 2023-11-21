# !/usr/bin/env python
#
# tidy3d documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import codecs
import datetime
import os
import re
import sys
import subprocess
import tidy3d

full_build = True

# TODO sort this out
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath("source"))
sys.path.insert(0, os.path.abspath("notebooks"))
sys.path.insert(0, os.path.abspath(""))
sys.path.insert(0, os.path.abspath("../tidy3d"))
sys.path.insert(0, os.path.abspath("../tidy3d/components"))
sys.path.insert(0, os.path.abspath("../tidy3d/web"))
sys.path.insert(0, os.path.abspath("../tidy3d/plugins"))
sys.path.append(os.path.abspath("_ext"))



# -- Project information -----------------------------------------------------

project = "Tidy3D"
author = "Flexcompute"
year = datetime.date.today().strftime("%Y")
copyright = f"Flexcompute 2020-{year}"
master_doc = "index"  # The master toctree document.s


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
add_module_names = False  # Remove namespaces from class/method signatures
autosummary_generate = full_build  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
## TODO DEBATE KEEP
autodoc_default_options = {"inherited-members": True, "show-inheritance":True}
# autodoc_pydantic_model_show_json = True
# autodoc_pydantic_settings_show_json = False
autodoc_pydantic_model_signature_prefix = "class"
autodoc_pydantic_field_signature_prefix = "attribute"
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
# autodoc_pydantic_model_members = False
# autodoc_pydantic_field_list_validators = False
# autodoc_pydantic_settings_summary_list_order = 'bysource'
# autodoc_pydantic_model_undoc_members = False
# autoclass_content = "class"
##
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
# custom_sitemap_excludes=[r'/notebooks/'] # TODO FIX
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
extensions = [
    # "custom-meta", # TODO FIX
    # "custom-sitemap", # TODO FIX
    # "custom-robots", # TODO FIX
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",  # Integrate Jupyter Notebooks and Sphinx
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx_copybutton",
    # "sphinx_sitemap", # TODO FIX
]
extlinks = {}
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
language='en'
latex_documents = [
    (master_doc, "tidy3d.tex", "tidy3d Documentation", "Dario Quintero", "manual"),
]
html_baseurl = "https://docs.flexcompute.com/projects/tidy3d/"
html_css_files = [
    "css/custom.css",
]
html_extra_path = ["./_static/robots.txt"]
html_favicon = "_static/logo.ico"
html_js_files = ["js/custom-download.js"]
htmlhelp_basename = "tidy3ddoc"
html_show_sourcelink = True  # Remove 'view source code' from top of page (for html, not python)
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_title = "Tidy3D Electromagnetic Solver"
html_theme_options = {
    "logo": {
        "image_light": "_static/img/Tidy3D-logo.svg",
        "image_dark": "_static/img/Tidy3D-logo-white.svg",
    },
    "path_to_docs": "docs",
    "repository_url": "https://github.com/flexcompute/tidy3d",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org/v2/gh/flexcompute-readthedocs/tidy3d-docs/readthedocs?labpath=docs%2Fsource%2Fnotebooks",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": False,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
latex_engine = "xelatex"
language = "en"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
man_pages = [(master_doc, "tidy3d", "tidy3d Documentation", [author], 1)]
mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}
nbsphinx_allow_errors = True  # Continue through Jupyter errors
nbsphinx_execute = "never"
project = "tidy3d"
pygments_style = "sphinx"
release = tidy3d.__version__
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# sitemap_url_scheme = "{lang}{version}{link}" # TODO FIX
source_suffix = [".rst", ".md"]
# templates_path = ["_templates"] # TODO evaluate
texinfo_documents = [
    (
        master_doc,
        "tidy3d",
        "tidy3d Documentation",
        author,
        "tidy3d",
        "One line description of project.",
        "Miscellaneous",
    ),
]
todo_include_todos = False



GIT_TAG_OUTPUT = subprocess.check_output(["git", "tag", "--points-at", "HEAD"])
GIT_BRANCH_OUTPUT = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
current_tag = GIT_TAG_OUTPUT.decode().strip()
current_branch = GIT_BRANCH_OUTPUT.decode().strip()
print(current_tag, current_branch)
if not current_tag and current_branch:
    if current_branch == "develop":
        version = "stable"
    elif current_branch == 'latest':
        version = "latest"
    else:
        version = "latest"
elif current_tag:
    if re.match(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$", current_tag):
        version = current_tag
    else:
        version = "latest"
# version = tidy3d.__version__

latex_elements: dict = {
    "preamble": r"\usepackage{bm}\n\usepackage{amssymb}\n\usepackage{esint}",
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
