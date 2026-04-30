import pathlib
from unittest.mock import MagicMock, patch

import fsspec
import numpy as np
import pynapple as nap
import pytest

import pynaviz as viz
from pynaviz.qt.references import EphysReference

has_ephys_reader = hasattr(nap, "EphysReader")
skip_no_ephys = pytest.mark.skipif(
    not has_ephys_reader, reason="requires pynapple >= 0.11 (EphysReader)"
)


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


@pytest.fixture
def shared_test_files(tmp_path_factory, dummy_tsd, dummy_tsdframe, dummy_tsdtensor):
    """Create test files that persist across multiple tests."""
    data_dir = tmp_path_factory.mktemp("data")
    test_dir = pathlib.Path(__file__).parent

    dummy_tsd.save(data_dir / "test_tsd.npz")
    dummy_tsdframe.save(data_dir / "test_tsd_frame.npz")
    dummy_tsdtensor.save(data_dir / "test_tsd_tensor.npz")
    video_file = test_dir / "test_video" / "numbered_video.mp4"
    video = viz.audiovideo.VideoHandler(video_file)

    # download nwb
    url = "https://osf.io/download/y7zwd/"
    nwb_file = test_dir / "test_nwb" / "nwb_file.nwb"
    download_osf_file_chunked(url, nwb_file)
    data = nap.load_file(nwb_file)

    expected_output = {
        "Tsd": dummy_tsd,
        "TsdFrame": dummy_tsdframe,
        "TsdTensor": dummy_tsdtensor,
        "VideoWidget": video,
        "dict": data,
    }
    return data_dir, video_file, nwb_file, expected_output


def test_load_files(shared_test_files, qtbot):
    """Test loading files directly via _load_multiple_files."""
    path_dir, video_path, nwb_file, expected = shared_test_files
    all_files = [*path_dir.iterdir(), video_path, nwb_file]

    main_window = viz.qt.mainwindow.MainWindow({})
    # dock = viz.qt.mainwindow.VariableWidget({}, main_window)
    main_window._load_multiple_files(all_files)

    for var in main_window.variables.values():
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
        elif name == "PosixPath":
            assert var == expected["VideoWidget"].file_path
        else:
            raise ValueError(f"Unknown variable: {name}")
    main_window.close()


@patch('pynaviz.qt.mainwindow.QFileDialog.getOpenFileNames')
def test_open_file_dialog(mock_dialog, shared_test_files, qtbot):
    """Test the open_file method with mocked dialog."""
    path_dir, video_file, nwb_file, expected = shared_test_files

    # print(path_dir)

    # Get list of test files
    test_files = [f.as_posix() for f in path_dir.iterdir()] + [video_file, nwb_file]

    # Mock dialog returns (list of files, selected filter)
    mock_dialog.return_value = (test_files, "")

    main_window = viz.qt.mainwindow.MainWindow({})
    main_window.open_file()

    # Verify dialog was called correctly
    mock_dialog.assert_called_once()

    # Verify files were loaded in the tree widget
    item_dict, flat_items = tree_widget_to_dict(main_window.variable_dock.treeWidget)
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
    for item in flat_items[0:3]:
        print(item.text(0))
        main_window.variable_dock.on_item_double_clicked(item, 0)
    main_window.close()


# ---------------------------------------------------------------------------
# open_folder (EphysReader / directory support)
# ---------------------------------------------------------------------------


@skip_no_ephys
@patch("pynaviz.qt.mainwindow.QFileDialog.getExistingDirectory")
def test_open_folder_calls_load(mock_dialog, tmp_path, qtbot):
    """open_folder passes the selected directory to _load_multiple_files."""
    rec_dir = tmp_path / "session"
    rec_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.keys.return_value = ["lfp", "mua"]
    mock_reader.name = "session"

    mock_dialog.return_value = str(rec_dir)

    main_window = viz.qt.mainwindow.MainWindow({})
    qtbot.addWidget(main_window)

    with patch("pynaviz.qt.mainwindow.nap.EphysReader", return_value=mock_reader):
        main_window.open_folder()

    mock_dialog.assert_called_once()
    assert "session" in main_window.variables
    inner = main_window.variables["session"]
    assert set(inner.keys()) == {"lfp", "mua"}
    for ref in inner.values():
        assert isinstance(ref, EphysReference)

    main_window.close()


@skip_no_ephys
@patch("pynaviz.qt.mainwindow.QFileDialog.getExistingDirectory")
def test_open_folder_cancelled(mock_dialog, tmp_path, qtbot):
    """open_folder does nothing when the dialog is cancelled (returns empty string)."""
    mock_dialog.return_value = ""

    main_window = viz.qt.mainwindow.MainWindow({})
    qtbot.addWidget(main_window)
    main_window.open_folder()

    assert main_window.variables == {}
    main_window.close()


@skip_no_ephys
def test_load_multiple_files_directory(tmp_path, qtbot):
    """_load_multiple_files loads a directory as an EphysReader tree."""
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.keys.return_value = ["ch0", "ch1"]
    mock_reader.name = "rec"

    main_window = viz.qt.mainwindow.MainWindow({})
    qtbot.addWidget(main_window)

    with patch("pynaviz.qt.mainwindow.nap.EphysReader", return_value=mock_reader):
        main_window._load_multiple_files([str(rec_dir)])

    assert "rec" in main_window.variables
    for ref in main_window.variables["rec"].values():
        assert isinstance(ref, EphysReference)

    main_window.close()


@skip_no_ephys
@pytest.mark.parametrize("ext", [".plx", ".ncs", ".smr"])
def test_load_multiple_files_ephys_extension(tmp_path, ext, qtbot):
    """_load_multiple_files loads known ephys file extensions via EphysReader."""
    ephys_file = tmp_path / f"recording{ext}"
    ephys_file.write_bytes(b"fake")

    mock_reader = MagicMock()
    mock_reader.keys.return_value = ["sig"]
    mock_reader.name = "recording"

    main_window = viz.qt.mainwindow.MainWindow({})
    qtbot.addWidget(main_window)

    with patch("pynaviz.qt.mainwindow.nap.EphysReader", return_value=mock_reader):
        main_window._load_multiple_files([str(ephys_file)])

    assert ephys_file.name in main_window.variables
    for ref in main_window.variables[ephys_file.name].values():
        assert isinstance(ref, EphysReference)

    main_window.close()


@skip_no_ephys
def test_load_multiple_files_ephys_error_skipped(tmp_path, qtbot, capsys):
    """If EphysReader raises on an ephys file, the file is skipped gracefully."""
    ephys_file = tmp_path / "bad.plx"
    ephys_file.write_bytes(b"fake")

    main_window = viz.qt.mainwindow.MainWindow({})
    qtbot.addWidget(main_window)

    with patch("pynaviz.qt.mainwindow.nap.EphysReader", side_effect=Exception("bad")):
        main_window._load_multiple_files([str(ephys_file)])

    assert main_window.variables == {}
    captured = capsys.readouterr()
    assert "EphysReader" in captured.out or "bad" in captured.out

    main_window.close()
