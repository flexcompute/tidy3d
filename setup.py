from setuptools import setup, find_packages

# with open("tidy3d/requirements.txt") as f:
#     required = f.read().splitlines()

with open("docs/source/notebooks/requirements.txt") as f:
    required = f.read().splitlines()
    
setup(
    packages=find_packages(exclude=('tests',)),
    install_requires = required
)
