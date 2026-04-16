"""
Tests for variable_loader: _filter_paths, _extract_name_value, and get_pynapple_variables.
EphysReader tests are skipped when pynapple < 0.11.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pynapple as nap
import pytest

from pynaviz.qt.references import EphysReference, NWBReference
from pynaviz.qt.variable_loader import (
    EPHYS_EXTENSIONS,
    _filter_paths,
    get_pynapple_variables,
)

has_ephys_reader = hasattr(nap, "EphysReader")
skip_no_ephys = pytest.mark.skipif(
    not has_ephys_reader, reason="requires pynapple >= 0.11 (EphysReader)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_ephys_reader(keys=("lfp", "spikes")):
    """Return a minimal EphysReader mock with the given keys."""
    mock = MagicMock()
    mock.keys.return_value = list(keys)
    mock.name = "recording"
    return mock


# ---------------------------------------------------------------------------
# _filter_paths — file types
# ---------------------------------------------------------------------------


def test_filter_paths_video(tmp_path):
    """Video files are returned as (base_name, path_string)."""
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        f = tmp_path / f"clip{ext}"
        f.write_bytes(b"fake")
        name, value = _filter_paths(str(f))
        assert name == f.name
        assert value == str(f)


def test_filter_paths_nwb(tmp_path, dummy_tsdframe):
    """NWB files are returned as (base_name, dict-of-NWBReferences)."""
    nwb_path = tmp_path / "data.nwb"
    mock_nwb = MagicMock(spec=nap.NWBFile)
    mock_nwb.keys.return_value = ["lfp", "spikes"]

    with patch("pynaviz.qt.variable_loader.nap.load_file", return_value=mock_nwb):
        name, value = _filter_paths(str(nwb_path))

    # load_file is called, but the file doesn't exist so we check via mock
    # Instead do a proper existence check: write a dummy file first
    nwb_path.write_bytes(b"fake")
    with patch("pynaviz.qt.variable_loader.nap.load_file", return_value=mock_nwb):
        name, value = _filter_paths(str(nwb_path))

    assert name == "data.nwb"
    assert set(value.keys()) == {"lfp", "spikes"}
    for ref in value.values():
        assert isinstance(ref, NWBReference)
        assert ref.nwb_file is mock_nwb


def test_filter_paths_npz(tmp_path, dummy_tsdframe):
    """NPZ files holding a pynapple object are returned as (base_name, obj)."""
    npz_path = tmp_path / "data.npz"
    dummy_tsdframe.save(npz_path)
    name, value = _filter_paths(str(npz_path))
    assert name == "data.npz"
    assert isinstance(value, nap.TsdFrame)


def test_filter_paths_unsupported_extension(tmp_path):
    """Unknown file extensions return (None, None)."""
    f = tmp_path / "data.csv"
    f.write_bytes(b"a,b,c")
    name, value = _filter_paths(str(f))
    assert name is None
    assert value is None


def test_filter_paths_nonexistent(tmp_path):
    """A path that does not exist returns (None, None)."""
    name, value = _filter_paths(str(tmp_path / "ghost.nwb"))
    assert name is None
    assert value is None


# ---------------------------------------------------------------------------
# _filter_paths — directory (EphysReader)
# ---------------------------------------------------------------------------


@skip_no_ephys
def test_filter_paths_directory_success(tmp_path):
    """Directories are passed to EphysReader; result wrapped as EphysReferences."""
    mock_reader = _make_mock_ephys_reader(["lfp", "mua"])
    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()

    with patch("pynaviz.qt.variable_loader.nap.EphysReader", return_value=mock_reader):
        name, value = _filter_paths(str(rec_dir))

    assert name == "recording"
    assert set(value.keys()) == {"lfp", "mua"}
    for ref in value.values():
        assert isinstance(ref, EphysReference)
        assert ref.ephys_reader is mock_reader


@skip_no_ephys
def test_filter_paths_directory_failure(tmp_path):
    """If EphysReader raises on a directory, (None, None) is returned."""
    rec_dir = tmp_path / "not_ephys"
    rec_dir.mkdir()

    with patch("pynaviz.qt.variable_loader.nap.EphysReader", side_effect=Exception("bad format")):
        name, value = _filter_paths(str(rec_dir))

    assert name is None
    assert value is None


@skip_no_ephys
def test_filter_paths_directory_with_format(tmp_path):
    """ephys_format is forwarded to EphysReader when loading a directory."""
    mock_reader = _make_mock_ephys_reader(["lfp"])
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()

    with patch("pynaviz.qt.variable_loader.nap.EphysReader", return_value=mock_reader) as mock_cls:
        _filter_paths(str(rec_dir), ephys_format="NeuroScopeIO")

    mock_cls.assert_called_once_with(str(rec_dir), format="NeuroScopeIO")


# ---------------------------------------------------------------------------
# _filter_paths — ephys file extensions
# ---------------------------------------------------------------------------


@skip_no_ephys
@pytest.mark.parametrize("ext", [".plx", ".ncs", ".smr", ".nev"])
def test_filter_paths_ephys_extension(tmp_path, ext):
    """Known ephys extensions are loaded via EphysReader."""
    mock_reader = _make_mock_ephys_reader(["ch0"])
    f = tmp_path / f"recording{ext}"
    f.write_bytes(b"fake")

    with patch("pynaviz.qt.variable_loader.nap.EphysReader", return_value=mock_reader):
        name, value = _filter_paths(str(f))

    assert name == f.name
    assert "ch0" in value
    assert isinstance(value["ch0"], EphysReference)


@skip_no_ephys
def test_filter_paths_explicit_format_overrides_extension(tmp_path):
    """An explicit ephys_format causes EphysReader to be tried even for unknown extensions."""
    mock_reader = _make_mock_ephys_reader(["sig"])
    f = tmp_path / "data.bin"
    f.write_bytes(b"fake")

    with patch("pynaviz.qt.variable_loader.nap.EphysReader", return_value=mock_reader) as mock_cls:
        name, value = _filter_paths(str(f), ephys_format="RawBinarySignalIO")

    mock_cls.assert_called_once_with(str(f), format="RawBinarySignalIO")
    assert name == "data.bin"
    assert isinstance(value["sig"], EphysReference)


# ---------------------------------------------------------------------------
# get_pynapple_variables — basic types
# ---------------------------------------------------------------------------


def test_get_pynapple_variables_none():
    assert get_pynapple_variables(None) == {}


def test_get_pynapple_variables_str_path(tmp_path, dummy_tsdframe):
    """A single string path is treated as a one-element list."""
    npz_path = tmp_path / "data.npz"
    dummy_tsdframe.save(npz_path)
    result = get_pynapple_variables(str(npz_path))
    assert "data.npz" in result
    assert isinstance(result["data.npz"], nap.TsdFrame)


def test_get_pynapple_variables_list_pynapple(dummy_tsd, dummy_tsdframe):
    result = get_pynapple_variables([dummy_tsd, dummy_tsdframe])
    assert "Tsd" in result
    assert "TsdFrame" in result


def test_get_pynapple_variables_duplicate_names(dummy_tsd):
    """Duplicate class names get numeric suffixes."""
    tsd2 = nap.Tsd(t=np.arange(10), d=np.random.randn(10))
    result = get_pynapple_variables([dummy_tsd, tsd2])
    assert "Tsd" in result
    assert "Tsd_1" in result


def test_get_pynapple_variables_dict(dummy_tsd, dummy_tsdframe):
    result = get_pynapple_variables({"lfp": dummy_tsdframe, "tsd": dummy_tsd})
    assert set(result.keys()) == {"lfp", "tsd"}


# ---------------------------------------------------------------------------
# get_pynapple_variables — NWBFile / EphysReader produce tree-view dicts
# ---------------------------------------------------------------------------


def test_get_pynapple_variables_nwbfile_is_nested():
    """NWBFile input → nested dict keyed by class name (tree-view consistency)."""
    mock_nwb = MagicMock(spec=nap.NWBFile)
    mock_nwb.keys.return_value = ["spikes", "lfp"]

    result = get_pynapple_variables(mock_nwb)
    # top-level key is the class name
    assert "NWBFile" in result
    inner = result["NWBFile"]
    assert set(inner.keys()) == {"spikes", "lfp"}
    for ref in inner.values():
        assert isinstance(ref, NWBReference)


@skip_no_ephys
def test_get_pynapple_variables_ephysreader_is_nested():
    """EphysReader input → nested dict keyed by recording name (tree-view consistency).

    Uses a minimal fake class so that isinstance(reader, nap.EphysReader) holds
    inside get_pynapple_variables when both nap.EphysReader and the reader are
    the same fake class.
    """
    class _FakeEphysReader:
        name = "session_2024"
        def keys(self):
            return ["ch0", "ch1"]

    reader = _FakeEphysReader()

    with patch("pynaviz.qt.variable_loader.nap.EphysReader", _FakeEphysReader):
        result = get_pynapple_variables(reader)

    assert "session_2024" in result
    inner = result["session_2024"]
    assert set(inner.keys()) == {"ch0", "ch1"}
    for ref in inner.values():
        assert isinstance(ref, EphysReference)


@skip_no_ephys
def test_get_pynapple_variables_ephys_format_forwarded(tmp_path):
    """ephys_format is forwarded to _filter_paths for every path in the list."""
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()

    # Patch _filter_paths to capture the call; avoid touching isinstance(x, nap.EphysReader)
    with patch("pynaviz.qt.variable_loader._filter_paths", return_value=("rec", {})) as mock_fp:
        get_pynapple_variables([str(rec_dir)], ephys_format="NeuroScopeIO")

    mock_fp.assert_called_once_with(str(rec_dir), ephys_format="NeuroScopeIO")


# ---------------------------------------------------------------------------
# EPHYS_EXTENSIONS constant
# ---------------------------------------------------------------------------


def test_ephys_extensions_are_lowercase_dotted():
    for ext in EPHYS_EXTENSIONS:
        assert ext.startswith(".")
        assert ext == ext.lower()
