import pynaviz as viz
import pytest
import pynapple as nap
import numpy as np
from PyQt6.QtWidgets import QApplication
import sys


@pytest.fixture(scope="module")  # or scope="session"
def shared_test_files(tmp_path_factory):
    """Create test files that persist across multiple tests."""
    tmp_dir = tmp_path_factory.mktemp("data")

    # Create your files
    time = np.arange(10)
    data = np.ones(10)
    tsd = nap.Tsd(time, data)
    tsd_frame = nap.TsdFrame(tsd.t, tsd.d[..., None])
    tsd_tensor = nap.TsdTensor(tsd_frame.t, tsd_frame.d[..., None])
    tsd.save(tmp_dir / "test_tsd.npz")
    tsd_frame.save(tmp_dir / "test_tsd_frame.npz")
    tsd_tensor.save(tmp_dir / "test_tsd_frame.npz")
    expected_output = {
        "Tsd": tsd,
        "TsdFrame": tsd_frame,
        "TsdTensor": tsd_tensor,
    }
    return tmp_dir, expected_output

def test_load_files(shared_test_files):
    path_dir, expected = shared_test_files
    # Set up Qt application (ensure it's headless for testing)
    if not QApplication.instance():
        app = QApplication(sys.argv)
        # Make headless for testing
        app.setQuitOnLastWindowClosed(False)
    else:
        app = QApplication.instance()

    main_window = viz.qt.mainwindow.MainWindow()
    dock = viz.qt.mainwindow.MainDock({}, main_window)
    main_window._load_multiple_files(path_dir.iterdir())
    for var in dock.variables.values():
        name = var.__class__.__name__
        if name in ["Tsd", "TsdFrame", "TsdTensor"]:
            np.testing.assert_array_equal(var.t, expected[name].t)
            np.testing.assert_array_equal(var.d, expected[name].d)
        elif name in ["NWBReference"]:
            assert var.key == expected[name].key
            assert var.nwb_file.path == expected[name].nwb_file.path




