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

with open("requirements.txt") as f:
    required = f.read().splitlines()

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
    # package_dir={"": ""},
    # packages=[PACKAGE_NAME],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=required,
)
