from __future__ import annotations

import os


def html_page_context(app, pagename, templatename, context, doctree):
    notebook_path = app.env.doc2path(os.path.abspath("" + pagename), base=None)
    if "/notebooks/" in notebook_path or notebook_path.endswith("examples.rst"):
        if "metatags" in context:
            context["metatags"] += "".join(['\n\t<meta content="noindex" name="robots" />'])


def setup(app):
    app.connect("html-page-context", html_page_context)
