from setuptools import setup, find_packages

# with open("tidy3d/requirements.txt") as f:
#     required = f.read().splitlines()

with open("requirements.txt") as f:
    required = f.read().splitlines()
    
import os
print(os.getcwd())

setup(
    packages=['../../../../tidy3d'],
    install_requires = required
)
