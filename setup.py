from setuptools import setup, find_packages

# with open("tidy3d/requirements.txt") as f:
#     required = f.read().splitlines()

# with open("docs/source/notebooks/requirements.txt") as f:
#     required = f.read().splitlines()
    
setup(
    packages=find_packages(),
    install_requires=[
        numpy,
        matplotlib,
        tmm,
        nlopt
        tqdm,
        gdspy,
    ],
)
