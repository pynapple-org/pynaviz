import sys
from pathlib import Path
from unittest.mock import patch

import pynapple as nap
import pytest
from PySide6.QtWidgets import QApplication

from pynaviz.cli import main

has_ephys_reader = hasattr(nap, "EphysReader")
skip_no_ephys = pytest.mark.skipif(
    not has_ephys_reader, reason="requires pynapple >= 0.11 (EphysReader)"
)


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


def test_cli_missing_file_exits(monkeypatch):
    """Non-existent file path causes a parser error (SystemExit with non-zero code)."""
    monkeypatch.setattr(sys, "argv", ["pynaviz", "/nonexistent/path/data.nwb"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


@skip_no_ephys
def test_cli_with_directory(monkeypatch, tmp_path):
    """A directory path is accepted by the CLI without error."""
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()

    # Patch _filter_paths so no real EphysReader I/O is attempted
    with patch("pynaviz.qt.variable_loader._filter_paths", return_value=("recording", {})):
        monkeypatch.setattr(sys, "argv", ["pynaviz", str(rec_dir)])
        main()

    assert QApplication.instance() is not None


@skip_no_ephys
def test_cli_format_flag_forwarded(monkeypatch, tmp_path):
    """--format value is forwarded through scope() down to _filter_paths."""
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()

    with patch("pynaviz.qt.variable_loader._filter_paths", return_value=("rec", {})) as mock_fp:
        monkeypatch.setattr(sys, "argv", ["pynaviz", str(rec_dir), "-f", "NeuroScopeIO"])
        main()

    mock_fp.assert_called_once_with(str(rec_dir), ephys_format="NeuroScopeIO")
