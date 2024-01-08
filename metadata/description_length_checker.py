import json

# Specify the metadata file name
metadata_file = "metadata.txt"

# Load the metadata dictionary from the text file
with open(metadata_file, "r", encoding="utf-8") as f:
    metadata_dict = json.loads(f.read())

# For each item in the metadata dictionary
for file_name, metadata in metadata_dict.items():
    description = metadata.get("description", "")
    description_length = len(description)

    if description_length > 150 or description_length < 70:
        print(f"{file_name}: {description_length}")
