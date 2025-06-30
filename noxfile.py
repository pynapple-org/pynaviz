import nox


# nox --no-venv -s linters
# nox --no-venv -s tests
# WGPU_FORCE_OFFSCREEN=1 nox

@nox.session(name="linters")
def linters(session):
    """Run linters"""
    session.run("ruff", "check", "src", "--ignore", "D")


@nox.session(name="tests")
def tests(session):
    """Run the test suite."""
    session.run("pytest", "--cov=src/uviz", "tests/", env={"WGPU_FORCE_OFFSCREEN": "1"})  # , "--pdb", "--pdbcls=IPython.terminal.debugger:Pdb")