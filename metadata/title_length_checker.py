import json

# Specify the metadata file name
metadata_file = 'metadata.txt'

# Load the metadata dictionary from the text file
with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata_dict = json.loads(f.read())

# For each item in the metadata dictionary
for file_name, metadata in metadata_dict.items():
    title = metadata.get('title', '')
    title_length = len(title)
    
    if title_length>60 or title_length<30:
        print(f'{file_name}: {title_length}')
