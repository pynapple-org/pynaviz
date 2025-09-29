import sys
from unittest.mock import patch

import numpy as np
import pynapple as nap
import pytest
from PyQt6.QtWidgets import QApplication, QDockWidget

import pynaviz as viz


def tree_widget_to_dict(tree_widget):
    """Extract all items from a QTreeWidget as a nested dictionary.

    Parameters
    ----------
    tree_widget : QTreeWidget
        The tree widget to extract items from.

    Returns
    -------
    dict
        Nested dictionary representing the tree structure.
    """

    def process_item(item):
        """Recursively process a tree item and its children."""
        # Get the item's text (assuming single column, adjust if needed)
        key = item.text(0)

        # If item has children, recurse
        child_count = item.childCount()
        if child_count > 0:
            children = {}
            for i in range(child_count):
                child = item.child(i)
                child_dict = process_item(child)
                children.update(child_dict)
            return {key: children}, []
        else:
            # Leaf node - you might want to store the actual data here
            # For now, just use the key
            return {key: None}, [item]  # or return {key: item.data(0, Qt.ItemDataRole.UserRole)}

    result = {}
    flat_items = []
    # Process all top-level items
    for i in range(tree_widget.topLevelItemCount()):
        item = tree_widget.topLevelItem(i)
        item_dict, flat_sub = process_item(item)
        result.update(item_dict)
        flat_items.extend(flat_sub)

    return result, flat_items


@pytest.fixture(scope="module")
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
    tsd_tensor.save(tmp_dir / "test_tsd_tensor.npz")
    expected_output = {
        "Tsd": tsd,
        "TsdFrame": tsd_frame,
        "TsdTensor": tsd_tensor,
    }
    return tmp_dir, expected_output


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication exists."""
    if not QApplication.instance():
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)
    else:
        app = QApplication.instance()
    return app


def test_load_files(shared_test_files, qapp):
    """Test loading files directly via _load_multiple_files."""
    path_dir, expected = shared_test_files

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


@patch('pynaviz.qt.mainwindow.QFileDialog.getOpenFileNames')
def test_open_file_dialog(mock_dialog, shared_test_files, qapp):
    """Test the open_file method with mocked dialog."""
    path_dir, expected = shared_test_files

    # Get list of test files
    test_files = [f.as_posix() for f in path_dir.iterdir()]

    # Mock dialog returns (list of files, selected filter)
    mock_dialog.return_value = (test_files, "")

    main_window = viz.qt.mainwindow.MainWindow()
    dock = viz.qt.mainwindow.MainDock({}, main_window)
    main_window.open_file()

    # Verify dialog was called correctly
    mock_dialog.assert_called_once()

    # Verify files were loaded in the tree widget
    item_dict, flat_items = tree_widget_to_dict(dock.treeWidget)
    assert  item_dict == {'test_tsd_frame.npz': None, 'test_tsd.npz': None, 'test_tsd_tensor.npz': None}

    # simulate dock creation
    for item in flat_items:
        dock.on_item_double_clicked(item, 0)

    children = main_window.findChildren(QDockWidget)
    children = [d.widget() for d in children if not isinstance(d, viz.qt.mainwindow.MainDock)]
    assert len(children) == len(expected)
    assert set(c.__class__.__name__ for c in children) == {"TsdWidget", "TsdFrameWidget", "TsdTensorWidget"}
