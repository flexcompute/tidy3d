import os
import json

# Specify the directory path
dir_path = os.path.join(os.path.dirname(__file__), '..')

# Specify the metadata file name
metadata_file = 'metadata.txt'

# Load the updated metadata dictionary from the text file
with open(metadata_file, 'r', encoding='utf-8') as f:
    updated_metadata = json.loads(f.read())

# For each .ipynb file
for file_name, metadata in updated_metadata.items():
    # Load the notebook
    notebook_file = os.path.join(dir_path, file_name)
    if os.path.exists(notebook_file):
        with open(notebook_file, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Update the metadata
        notebook['metadata']['title'] = metadata['title']
        notebook['metadata']['description'] = metadata['description']
        notebook['metadata']['keywords'] = metadata['keywords']
        notebook['metadata']['feature_image'] = metadata['feature_image']

        # Save the updated notebook
        with open(notebook_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
