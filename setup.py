import setuptools
from distutils.util import convert_path


PACKAGE_NAME = "tidy3d"
PIP_NAME = "tidy3d-beta"
REPO_NAME = "tidy3d"

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
core_required = read_requirements("requirements/core.txt")
core_required += basic_required + web_required
dev_required = read_requirements("requirements/dev.txt")

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
    python_requires=">=3.7",
    install_requires=core_required,
    extras_require={"dev": dev_required},
)
