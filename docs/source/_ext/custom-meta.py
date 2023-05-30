import os

def html_page_context(app, pagename, templatename, context, doctree):
    notebook_path = app.env.doc2path(os.path.abspath("source/" +pagename), base=None)

    if notebook_path.endswith('.ipynb'):
        context['metatags'] += "".join(['\n\t<meta content="noindex" name="robots" />'])

def setup(app):
    app.connect('html-page-context', html_page_context)
