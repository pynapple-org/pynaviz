import pathlib
import sys
from unittest.mock import patch

import fsspec
import numpy as np
import pynapple as nap
import pytest
from PyQt6.QtWidgets import QApplication, QDockWidget

import pynaviz as viz


def download_osf_file_chunked(url, output_path, chunk_size=8192):
    """Download a file from OSF in chunks.

    Parameters
    ----------
    url : str
        The OSF download URL
    output_path : str or Path
        Where to save the downloaded file
    chunk_size : int
        Size of chunks to read at a time (in bytes)
    """
    output_path = pathlib.Path(output_path)
    if output_path.exists():
        return
    output_path.parent.mkdir(exist_ok=True)
    with fsspec.open(url, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
    print(f"Downloaded to {output_path}")


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
            flat_items = []
            for i in range(child_count):
                child = item.child(i)
                child_dict, child_flat = process_item(child)  # Unpack both return values
                children.update(child_dict)
                flat_items.extend(child_flat)
            return {key: children}, flat_items
        else:
            # Leaf node
            return {key: None}, [item]

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
    data_dir = tmp_path_factory.mktemp("data")
    test_dir = pathlib.Path(__file__).parent

    # Create your files
    time = np.arange(10)
    data = np.ones(10)
    tsd = nap.Tsd(time, data)
    tsd_frame = nap.TsdFrame(tsd.t, tsd.d[..., None])
    tsd_tensor = nap.TsdTensor(tsd_frame.t, tsd_frame.d[..., None])
    tsd.save(data_dir / "test_tsd.npz")
    tsd_frame.save(data_dir / "test_tsd_frame.npz")
    tsd_tensor.save(data_dir / "test_tsd_tensor.npz")
    video_file = test_dir / "test_video" / "numbered_video.mp4"
    video = viz.audiovideo.VideoHandler(video_file)

    # download nwb
    url = "https://osf.io/download/y7zwd/"
    nwb_file = test_dir / "test_nwb" / "nwb_file.nwb"
    download_osf_file_chunked(url, nwb_file)
    data = nap.load_file(nwb_file)
    expected_output = {
        "Tsd": tsd,
        "TsdFrame": tsd_frame,
        "TsdTensor": tsd_tensor,
        "VideoWidget": video,
        "dict": data,
    }
    return data_dir, video_file, nwb_file, expected_output


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
    path_dir, video_path, nwb_file,  expected = shared_test_files
    all_files = (*path_dir.iterdir(), video_path, nwb_file)

    main_window = viz.qt.mainwindow.MainWindow()
    dock = viz.qt.mainwindow.MainDock({}, main_window)
    main_window._load_multiple_files(all_files)

    for var in dock.variables.values():
        name = var.__class__.__name__
        if name in ["Tsd", "TsdFrame", "TsdTensor"]:
            np.testing.assert_array_equal(var.t, expected[name].t)
            np.testing.assert_array_equal(var.d, expected[name].d)
        elif name == "NWBReference":
            assert var.key == expected[name].key
            assert var.nwb_file.path == expected[name].nwb_file.path
        elif name == "VideoWidget":
            assert isinstance(var.plot.data, expected[name].__class__)
            assert var.plot.data.file_path == expected[name].file_path
            np.testing.assert_array_equal(var.plot.data.get(1), expected[name].get(1))
        # NWB are dicts.
        elif name == "dict":
            for val in var.values():
                assert isinstance(val, viz.qt.mainwindow.NWBReference)
                assert val.nwb_file.path == expected[name].path
                assert isinstance(val.nwb_file, expected[name].__class__)
        else:
            raise ValueError(f"Unknown variable: {name}")


@patch('pynaviz.qt.mainwindow.QFileDialog.getOpenFileNames')
def test_open_file_dialog(mock_dialog, shared_test_files, qapp):
    """Test the open_file method with mocked dialog."""
    path_dir, video_file, nwb_file, expected = shared_test_files

    # Get list of test files
    test_files = [f.as_posix() for f in path_dir.iterdir()] + [video_file, nwb_file]

    # Mock dialog returns (list of files, selected filter)
    mock_dialog.return_value = (test_files, "")

    main_window = viz.qt.mainwindow.MainWindow()
    dock = viz.qt.mainwindow.MainDock({}, main_window)
    main_window.open_file()

    # Verify dialog was called correctly
    mock_dialog.assert_called_once()

    # Verify files were loaded in the tree widget
    item_dict, flat_items = tree_widget_to_dict(dock.treeWidget)
    assert  item_dict == {
        'numbered_video.mp4': None,
        'nwb_file.nwb': {'epochs': None,
                      'position_time_support': None,
                      'rx': None,
                      'ry': None,
                      'rz': None,
                      'units': None,
                      'x': None,
                      'y': None,
                      'z': None},
        'test_tsd.npz': None,
        'test_tsd_frame.npz': None,
        'test_tsd_tensor.npz': None
    }

    # simulate dock creation
    for item in flat_items:
        dock.on_item_double_clicked(item, 0)

    children = main_window.findChildren(QDockWidget)
    children = [d.widget() for d in children if not isinstance(d, viz.qt.mainwindow.MainDock)]
    assert len(children) == len(flat_items)
    print(set(c.__class__.__name__ for c in children))
    assert set(c.__class__.__name__ for c in children) == {"TsGroupWidget", "TsdWidget", "TsdFrameWidget", "TsdTensorWidget", "VideoWidget", "IntervalSetWidget"}
