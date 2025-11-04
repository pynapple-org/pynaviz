import sys

import pytest

import pynaviz


# -----------------------------
# Test 1: basic import works without Qt
# -----------------------------
def test_basic_import():
    # Access non-Qt features
    assert hasattr(pynaviz, "PlotTsd")
    assert hasattr(pynaviz, "PlotVideo")

# -----------------------------
# Test 2: lazy Qt import raises ImportError if Qt is missing
# -----------------------------
def test_lazy_qt_import_raises(monkeypatch):
    # simulate Qt not installed
    # temporarily remove PySide6 if present
    monkeypatch.setitem(sys.modules, "pynaviz.qt", None)

    with pytest.raises(ImportError) as excinfo:
        _ = pynaviz.TsdWidget  # triggers lazy import

    assert "Qt support is not installed" in str(excinfo.value)
