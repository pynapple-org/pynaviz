"""Tests for interval_sets_selection module.

Test structure mirrors test_tsdframe_selection.py pattern.
"""
import pytest
from collections import OrderedDict
import numpy as np
import pynapple as nap
from PyQt6.QtCore import Qt

from pynaviz.qt.interval_sets_selection import (
    IntervalSetsModel,
    ComboDelegate,
    SpinDelegate,
    IntervalSetsDialog,
)
from pynaviz.utils import GRADED_COLOR_LIST


# ========== Test Configuration Constants ==========
# These mirror the structure from test_tsdframe_selection.py

COLUMNS_IDS = OrderedDict({
    "name": 0,
    "colors": 1,
    "alpha": 2,
})

KEYS = set(COLUMNS_IDS.keys())
KEYS.add("checked")  # Extra key not in columns but in row data

KEY_TYPES = {
    "name": str,
    "colors": str,
    "alpha": (int, float),
    "checked": bool,
}

KEY_DEFAULTS = {
    "name": None,  # Will be set from dict keys
    "colors": None,  # Will be assigned from GRADED_COLOR_LIST
    "alpha": 0.5,
    "checked": False,
}

EXPECTED_HEADERS = ["Interval Set", "Color", "Alpha"]
EXPECTED_COLUMN_COUNT = len(COLUMNS_IDS)
EXPECTED_ROW_KEYS_COUNT = len(KEYS)

# Flag configurations for each column
COLUMN_FLAGS = {
    0: Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable,
    1: Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable,
    2: Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable,
}


# ========== Fixtures ==========

@pytest.fixture
def sample_interval_sets():
    """Create sample interval_sets dict for testing with real IntervalSet objects."""
    return {
        "sleep": nap.IntervalSet(
            start=[0, 100, 200],
            end=[50, 150, 250]
        ),
        "wake": nap.IntervalSet(
            start=[50, 150],
            end=[100, 200]
        ),
        "rem": nap.IntervalSet(
            start=[10, 110, 210],
            end=[30, 130, 230]
        ),
    }


@pytest.fixture
def empty_interval_sets():
    """Empty interval sets dict."""
    return {}


# ========== Test IntervalSetsModel ==========

class TestIntervalSetsModel:
    """Test suite for IntervalSetsModel."""

    @pytest.fixture
    def model(self, sample_interval_sets):
        """Create an IntervalSetsModel instance."""
        return IntervalSetsModel(sample_interval_sets)

    @pytest.fixture
    def empty_model(self, empty_interval_sets):
        """Create an IntervalSetsModel instance with empty data."""
        return IntervalSetsModel(empty_interval_sets)

    # ========== TEST 1: Initialization ==========

    def test_initialization_rows_count(self, model, sample_interval_sets):
        """
        Verify model creates correct number of rows.

        This checks:
        - len(model.rows) matches number of interval sets
        - rowCount() returns correct value
        """
        expected_count = len(sample_interval_sets)

        assert len(model.rows) == expected_count, (
            f"Expected {expected_count} rows, found {len(model.rows)}"
        )

        assert model.rowCount() == expected_count, (
            f"rowCount() returned {model.rowCount()}, expected {expected_count}"
        )

    def test_initialization_row_structure(self, model, sample_interval_sets):
        """
        Verify each row has correct structure and data types.

        This checks:
        - All expected keys are present (name, colors, alpha, checked)
        - Data types are correct
        - Default values are set properly
        - Name matches interval set key
        - Color is assigned from GRADED_COLOR_LIST
        """
        for i, (key_interval, row) in enumerate(zip(sample_interval_sets.keys(), model.rows)):
            # Check all expected keys are present
            assert set(row.keys()) == KEYS, (
                f"Row {i}: Unexpected key(s) found. "
                f"Difference: {KEYS.symmetric_difference(row.keys())}"
            )

            # Check data types
            for key, key_type in KEY_TYPES.items():
                assert isinstance(row[key], key_type), (
                    f"Row {i}, key '{key}': Expected type {key_type}, found {type(row[key])}"
                )

            # Check specific default values
            assert row["alpha"] == KEY_DEFAULTS["alpha"], (
                f"Row {i}: Expected alpha={KEY_DEFAULTS['alpha']}, found {row['alpha']}"
            )
            assert row["checked"] == KEY_DEFAULTS["checked"], (
                f"Row {i}: Expected checked={KEY_DEFAULTS['checked']}, found {row['checked']}"
            )

            # Check name matches interval set key
            assert row["name"] == key_interval, (
                f"Row {i}: Expected name '{key_interval}', found '{row['name']}'"
            )

            # Check color assignment from GRADED_COLOR_LIST
            expected_color = GRADED_COLOR_LIST[i % len(GRADED_COLOR_LIST)]
            assert row["colors"] == expected_color, (
                f"Row {i}: Expected color '{expected_color}', found '{row['colors']}'"
            )

    def test_column_count(self, model):
        """
        Verify column count is correct.
        """
        assert model.columnCount() == EXPECTED_COLUMN_COUNT, (
            f"Expected {EXPECTED_COLUMN_COUNT} columns, found {model.columnCount()}"
        )

    # ========== TEST 2: Header Data ==========

    @pytest.fixture
    def all_item_data_roles(self):
        """Get all ItemDataRole enum values."""
        return [
            getattr(Qt.ItemDataRole, r)
            for r in dir(Qt.ItemDataRole)
            if r.endswith("Role")
        ]

    def test_header_data(self, model):
        """
        Verify header labels are correct for all columns.

        Expected headers: ["Interval Set", "Color", "Alpha"]
        """
        for col, expected in enumerate(EXPECTED_HEADERS):
            header = model.headerData(
                col,
                Qt.Orientation.Horizontal,
                Qt.ItemDataRole.DisplayRole
            )
            assert header == expected, (
                f"Column {col}: Expected header '{expected}', found '{header}'"
            )

    def test_header_data_unsupported_roles(self, model, all_item_data_roles):
        """
        Verify header returns None for unsupported roles.

        Only DisplayRole is supported for headers.
        """
        supported_roles = {Qt.ItemDataRole.DisplayRole}
        unsupported_roles = [
            role for role in all_item_data_roles
            if role not in supported_roles
        ]

        for col in range(EXPECTED_COLUMN_COUNT):
            for role in unsupported_roles:
                result = model.headerData(col, Qt.Orientation.Horizontal, role)
                assert result is None, (
                    f"Column {col}, Role {role}: Expected None, found {result}"
                )

    def test_header_data_vertical_orientation(self, model, all_item_data_roles):
        """
        Verify vertical headers return None (not implemented).
        """
        for col in range(EXPECTED_COLUMN_COUNT):
            for role in all_item_data_roles:
                result = model.headerData(col, Qt.Orientation.Vertical, role)
                assert result is None, (
                    f"Vertical header for column {col}, role {role}: "
                    f"Expected None, found {result}"
                )

    def test_data_display_role_by_columns(self, model):
        """
        Verify data() returns correct values for DisplayRole.

        Tests all columns across all rows.
        """
        for row in range(model.rowCount()):
            for col, col_name in enumerate(COLUMNS_IDS.keys()):
                index = model.index(row, col)
                data = model.data(index, Qt.ItemDataRole.DisplayRole)
                expected = model.rows[row][col_name]
                assert data == expected, (
                    f"Row {row}, Column {col} ('{col_name}'): "
                    f"Expected {expected}, found {data}"
                )

    def test_data_edit_role_matches_display(self, model):
        """
        Verify EditRole returns same as DisplayRole for all cells.
        """
        for row in range(model.rowCount()):
            for col in range(model.columnCount()):
                index = model.index(row, col)
                display = model.data(index, Qt.ItemDataRole.DisplayRole)
                edit = model.data(index, Qt.ItemDataRole.EditRole)
                assert display == edit, (
                    f"Row {row}, Column {col}: DisplayRole ({display}) != EditRole ({edit})"
                )

    def test_data_checkstate_role_column_0_all_rows(self, model):
        """
        Verify CheckStateRole for column 0 across all rows.

        Column 0 should support checkbox state.
        """
        for row in range(model.rowCount()):
            index = model.index(row, 0)

            # Test unchecked state (default)
            checkstate = model.data(index, Qt.ItemDataRole.CheckStateRole)
            assert checkstate == Qt.CheckState.Unchecked, (
                f"Row {row}: Expected Unchecked, found {checkstate}"
            )

            # Modify and test checked state
            model.rows[row]["checked"] = True
            checkstate = model.data(index, Qt.ItemDataRole.CheckStateRole)
            assert checkstate == Qt.CheckState.Checked, (
                f"Row {row}: Expected Checked after modification, found {checkstate}"
            )

            # Reset for other tests
            model.rows[row]["checked"] = False

    def test_data_checkstate_role_other_columns(self, model):
        """
        Verify CheckStateRole returns None for columns 1-2.

        Only column 0 should support checkbox.
        """
        for row in range(model.rowCount()):
            for col in range(1, EXPECTED_COLUMN_COUNT):
                index = model.index(row, col)
                checkstate = model.data(index, Qt.ItemDataRole.CheckStateRole)
                assert checkstate is None, (
                    f"Row {row}, Column {col}: CheckStateRole should return None, "
                    f"found {checkstate}"
                )

    def test_data_unsupported_roles(self, model, all_item_data_roles):
        """
        Verify data() returns None for unsupported roles.

        Only DisplayRole, EditRole, and CheckStateRole are supported.
        """
        supported_roles = {
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.EditRole,
            Qt.ItemDataRole.CheckStateRole
        }
        unsupported_roles = [
            role for role in all_item_data_roles
            if role not in supported_roles
        ]

        for row in range(model.rowCount()):
            for col in range(model.columnCount()):
                index = model.index(row, col)
                for role in unsupported_roles:
                    data = model.data(index, role)
                    assert data is None, (
                        f"Row {row}, Column {col}, Role {role}: "
                        f"Expected None for unsupported role, found {data}"
                    )
