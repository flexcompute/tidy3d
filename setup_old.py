import setuptools
from distutils.util import convert_path
from setuptools.command.install import install
import os
from os.path import expanduser

TIDY3D_DIR = f"{expanduser('~')}/.tidy3d"
PACKAGE_NAME = "tidy3d"
REPO_NAME = "tidy3d"
PIP_NAME = "tidy3d"

version = {}
version_path = convert_path(f"{PACKAGE_NAME}/version.py")
with open(version_path) as version_file:
    exec(version_file.read(), version)

print(version["__version__"])

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def read_requirements(req_file: str):
    """Read requirements from a file excluding lines that have ``-r``."""
    with open(req_file) as f:
        required = f.read().splitlines()
    return [req for req in required if "-r" not in req]


basic_required = read_requirements("requirements/basic.txt")
web_required = read_requirements("requirements/web.txt")
jax_required = read_requirements("requirements/jax.txt")
gdstk_required = read_requirements("requirements/gdstk.txt")
gdspy_required = read_requirements("requirements/gdspy.txt")
trimesh_required = read_requirements("requirements/trimesh.txt")
core_required = read_requirements("requirements/core.txt")
core_required += basic_required + web_required
dev_required = read_requirements("requirements/dev.txt")


def create_config_folder():
    if not os.path.exists(TIDY3D_DIR):
        os.mkdir(TIDY3D_DIR)


create_config_folder()

setuptools.setup(
    name=PIP_NAME,
    version=version["__version__"],
    author="Tyler Hughes",
    author_email="tyler@flexcompute.com",
    description="A fast FDTD solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/flexcompute/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/flexcompute/{REPO_NAME}/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=core_required,
    extras_require={
        "dev": dev_required,
        "jax": jax_required,
        "gdspy": gdspy_required,
        "gdstk": gdstk_required,
        "trimesh": trimesh_required,
    },
    entry_points={
        "console_scripts": [
            "tidy3d = tidy3d.web.cli:tidy3d_cli",
        ],
    },
)
