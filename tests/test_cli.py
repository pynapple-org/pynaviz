import sys

import pytest
from PyQt6.QtWidgets import QApplication

from pynaviz.cli import main


@pytest.fixture(autouse=True)
def no_block_qt(monkeypatch):
    """
    Prevent QApplication.exec from blocking during tests.
    """
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)


def test_cli_no_files(monkeypatch, qtbot):
    """
    Test running pynaviz with no input files (should launch empty viewer).
    """
    monkeypatch.setattr(sys, "argv", ["pynaviz"])
    main()

    app = QApplication.instance()
    assert app is not None  # QApplication should be created


def test_cli_with_npz(monkeypatch, qtbot):
    """
    Test running pynaviz with a .npz file as input.
    """

    # Patch CLI args
    monkeypatch.setattr(sys, "argv", ["pynaviz", "filetest/tsdframe_minfo.npz"])
    main()

    app = QApplication.instance()
    assert app is not None  # QApplication should exist

    # You can later extend this to check that scope() was called with the right args


def test_cli_with_layout_and_files(monkeypatch, qtbot):
    """
    Test running pynaviz with a layout.json and multiple files.
    """

    monkeypatch.setattr(
        sys,
        "argv",
        ["pynaviz", "-l", "filetest/layout.json", "filetest/tsdframe_minfo.npz", "filetest/A2929-200711.nwb"]
    )
    main()

    app = QApplication.instance()
    assert app is not None
