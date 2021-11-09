rm -rf _autosummary
sphinx-build -b html . _build

# jupyter nbconvert --output-dir='.' --to script ../notebooks/StartHere.ipynb --no-prompt
