import os
import shutil
from pathlib import Path

import nox


# nox --no-venv -s linters
# nox --no-venv -s tests


@nox.session(name="linters")
def linters(session):
    """Run linters"""
    session.run("ruff", "check", "src", "--ignore", "D")


@nox.session(name="tests")
def tests(session):
    """Run the test suite."""
    session.run("pytest")  # , "--pdb", "--pdbcls=IPython.terminal.debugger:Pdb")
