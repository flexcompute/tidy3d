def html_page_context(app, pagename, templatename, context, doctree):
    notebook_path = app.env.doc2path(pagename, base=None)
    if "notebook" in notebook_path or notebook_path.endswith("examples.rst"):
        context["metatags"] = (
            context.get("metatags", "") + '\n\t<meta content="noindex" name="robots" />'
        )


def setup(app):
    app.connect("html-page-context", html_page_context)
