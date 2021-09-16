import nbconvert
import nbformat

import os
import sys
sys.path.append('./')

# load the notebook executor
from nbconvert.preprocessors import ExecutePreprocessor
ep = ExecutePreprocessor(timeout=1000)

# get all notebook files
notebook_directory = 'notebooks/'
notebook_filenames = [notebook_directory + f for f in os.listdir(notebook_directory) if '.ipynb' in f and f != '.ipynb_checkpoints']

def test_notebooks():

	# loop through notebooks and run each of them
	for fname in notebook_filenames:
		with open(fname) as f:
			nb = nbformat.read(f, as_version=4)

			# specify we are running the notebook from this path, so tidy3d is correctly imported
			ep.preprocess(nb, {'metadata': {'path': '.'}})
