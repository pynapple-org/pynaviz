from pathlib import Path

from setuptools import find_packages, setup

install_requires = [
    "fastplotlib",
    "pynapple",
]

extras_require = {
    "notebook": [
        "fastplotlib[notebook]"
    ],
    "docs": [

    ],
    "tests": [
        "pytest",
        "black"
    ]
}

with open(Path(__file__).parent.joinpath("README.md")) as f:
    readme = f.read()

with open(Path(__file__).parent.joinpath("pynaviz", "VERSION"), "r") as f:
    ver = f.read().split("\n")[0]

classifiers = [
    "Programming Language :: Python :: 3",
    "Topic :: Neuroscience :: Visualization",
    "License :: OSI Approved :: GNU Software License",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
]

authors = "Guillaume Viejo, Edoardo Balzani, Kushal Kolar, Caitlin Lewis"

setup(
    name="pynaviz",
    version=ver,
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://github.com/pynapple-org/pynaviz",
    license="GNU Lesser General Public License v3",
    author=authors,
    author_email="",
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    description="A neuroscience visualization library using pynapple and fastplotlib"
)
