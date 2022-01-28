rm -rf source/_autosummary
rm -rf source/_build

sphinx-build -b html source source/_build

jupyter nbconvert --output-dir='.' --to script source/notebooks/StartHere.ipynb --no-prompt
