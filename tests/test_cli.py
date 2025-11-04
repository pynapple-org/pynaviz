import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

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
    here = Path(__file__).parent
    npz_path = here / "filetest" / "tsdframe_minfo.npz"

    # Patch CLI args
    monkeypatch.setattr(sys, "argv", ["pynaviz", str(npz_path)])
    main()

    app = QApplication.instance()
    assert app is not None  # QApplication should exist

    # You can later extend this to check that scope() was called with the right args


def test_cli_with_layout_and_files(monkeypatch, qtbot):
    """
    Test running pynaviz with a layout.json and multiple files.
    """

    here = Path(__file__).parent
    layout_path = here / "filetest" / "layout.json"
    npz_path = here / "filetest" / "tsdframe_minfo.npz"
    # nwb_path = here / "filetest" / "A2929-200711.nwb"

    monkeypatch.setattr(
        sys,
        "argv",
        ["pynaviz", "-l", str(layout_path), str(npz_path)]
    )
    try:
        main()
    except SystemExit as e:
        # Catch CLI exit (status 0 means success)
        assert e.code == 0

    app = QApplication.instance()
    assert app is not None
