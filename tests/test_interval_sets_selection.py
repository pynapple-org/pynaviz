"""Tests for interval_sets_selection module.

Test structure mirrors test_tsdframe_selection.py pattern.
"""
from collections import OrderedDict

import pynapple as nap
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QPushButton

from pynaviz.qt.interval_sets_selection import (
    ComboDelegate,
    IntervalSetsDialog,
    IntervalSetsModel,
    SpinDelegate,
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

def get_column_index(header_name: str) -> int:
    """Get column index from header name."""
    return EXPECTED_HEADERS.index(header_name)


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

    def test_flags_all_columns(self, model):
        """
        Verify flags for all columns across all rows.

        Column 0: Enabled + Selectable + UserCheckable
        Column 1: Enabled + Selectable + Editable
        Column 2: Enabled + Selectable + Editable
        """
        for row in range(model.rowCount()):
            for col, expected_flags in COLUMN_FLAGS.items():
                index = model.index(row, col)
                flags = model.flags(index)
                assert flags == expected_flags, (
                    f"Row {row}, Column {col}: Flag mismatch.\n"
                    f"Expected: {expected_flags}\n"
                    f"Found: {flags}"
                )

    def test_flags_base_always_present(self, model):
        """
        Verify base flags are always present.

        All columns should have Enabled and Selectable flags.
        """
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

        for row in range(model.rowCount()):
            for col in range(model.columnCount()):
                index = model.index(row, col)
                flags = model.flags(index)
                assert (flags & base) == base, (
                    f"Row {row}, Column {col}: Base flags missing. Found flags: {flags}"
                )

    @pytest.mark.parametrize("check_state,expected_bool", [
        (Qt.CheckState.Checked, True),
        (Qt.CheckState.Unchecked, False),
    ])
    def test_setdata_checkstate(self, model, qtbot, check_state, expected_bool):
        """
        Verify setting check state for column 0.

        Tests both Checked and Unchecked states across all rows.
        Signal should emit: (name, colors, alpha, checked)
        """
        for row in range(model.rowCount()):
            index = model.index(row, 0)

            with qtbot.waitSignal(model.checkStateChanged) as blocker:
                result = model.setData(index, check_state, Qt.ItemDataRole.CheckStateRole)

            assert result is True, (
                f"Row {row}: setData returned False for CheckStateRole"
            )
            assert model.rows[row]["checked"] is expected_bool, (
                f"Row {row}: Expected checked={expected_bool}, found {model.rows[row]['checked']}"
            )

            # Verify signal arguments: (name, colors, alpha, checked)
            assert blocker.args[0] == model.rows[row]["name"], (
                f"Row {row}: Signal arg[0] (name) mismatch"
            )
            assert blocker.args[1] == model.rows[row]["colors"], (
                f"Row {row}: Signal arg[1] (colors) mismatch"
            )
            assert blocker.args[2] == model.rows[row]["alpha"], (
                f"Row {row}: Signal arg[2] (alpha) mismatch"
            )
            assert blocker.args[3] is expected_bool, (
                f"Row {row}: Signal arg[3] (checked) expected {expected_bool}, found {blocker.args[3]}"
            )

    @pytest.mark.parametrize("col,key,test_values", [
        (1, "colors", ["red", "blue", "#FF5733"]),
        (2, "alpha", [0.0, 0.5, 1.0]),
    ])
    def test_setdata_edit_role(self, model, qtbot, col, key, test_values):
        """
        Verify editing values in editable columns.

        Column 1: colors (string)
        Column 2: alpha (float 0.0-1.0)
        """
        for row in range(model.rowCount()):
            for test_value in test_values:
                index = model.index(row, col)

                with qtbot.waitSignal(model.checkStateChanged) as blocker:
                    result = model.setData(index, test_value, Qt.ItemDataRole.EditRole)

                assert result is True, (
                    f"Row {row}, Column {col} ('{key}'): "
                    f"setData returned False for value {test_value}"
                )

                # Verify data was stored correctly
                expected_value = str(test_value) if key == "colors" else float(test_value)
                assert model.rows[row][key] == expected_value, (
                    f"Row {row}, Column {col} ('{key}'): "
                    f"Expected {expected_value}, found {model.rows[row][key]}"
                )

                # Verify signal contains correct value
                # Signal args: (name, colors, alpha, checked)
                signal_index = col  # col 1 -> arg[1], col 2 -> arg[2]
                assert blocker.args[signal_index] == expected_value, (
                    f"Row {row}, Column {col} ('{key}'): "
                    f"Signal arg[{signal_index}] expected {expected_value}, "
                    f"found {blocker.args[signal_index]}"
                )

    def test_setdata_edit_role_column_0_fails(self, model):
        """
        Verify editing column 0 with EditRole fails.

        Column 0 should only support CheckStateRole, not EditRole.
        """
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            original_name = model.rows[row]["name"]
            result = model.setData(index, "new_name", Qt.ItemDataRole.EditRole)

            assert result is False, (
                f"Row {row}: setData should return False for editing column 0"
            )
            assert model.rows[row]["name"] == original_name, (
                f"Row {row}: Name should not change. Expected '{original_name}', "
                f"found '{model.rows[row]['name']}'"
            )

    def test_setdata_unsupported_roles(self, model, all_item_data_roles):
        """
        Verify setData returns False for unsupported roles.
        """
        unsupported_roles = [
            role for role in all_item_data_roles
            if role not in {Qt.ItemDataRole.EditRole, Qt.ItemDataRole.CheckStateRole}
        ]

        # Test first 2 rows for efficiency
        for row in range(min(2, model.rowCount())):
            for col in range(model.columnCount()):
                for role in unsupported_roles:
                    index = model.index(row, col)
                    result = model.setData(index, "test_value", role)
                    assert result is False, (
                        f"Row {row}, Column {col}, Role {role}: "
                        f"setData should return False for unsupported role"
                    )

    def test_setdata_emits_datachanged(self, model, qtbot):
        """
        Verify setData emits dataChanged signal.
        """
        for row in range(model.rowCount()):
            index = model.index(row, 1)

            with qtbot.waitSignal(model.dataChanged) as blocker:
                model.setData(index, "blue", Qt.ItemDataRole.EditRole)

            assert blocker.args[0] == index, (
                f"Row {row}: dataChanged signal arg[0] (start index) mismatch"
            )
            assert blocker.args[1] == index, (
                f"Row {row}: dataChanged signal arg[1] (end index) mismatch"
            )

    def test_multiple_rows_independence(self, model):
        """
        Verify modifying one row doesn't affect others.
        """
        if model.rowCount() < 2:
            pytest.skip("Need at least 2 rows for this test")

        # Store original values for all rows except first
        original_values = [
            {
                "checked": model.rows[row]["checked"],
                "colors": model.rows[row]["colors"],
                "alpha": model.rows[row]["alpha"],
            }
            for row in range(1, model.rowCount())
        ]

        # Modify first row
        model.setData(model.index(0, get_column_index("Interval Set")), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
        model.setData(model.index(0, get_column_index("Color")), "modified_color", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, get_column_index("Alpha")), 0.99, Qt.ItemDataRole.EditRole)

        # Check other rows are unaffected
        for i, row in enumerate(range(1, model.rowCount())):
            for key, original_value in original_values[i].items():
                current_value = model.rows[row][key]
                assert current_value == original_value, (
                    f"Row {row}, key '{key}': Value changed unexpectedly. "
                    f"Expected {original_value}, found {current_value}"
                )

    def test_empty_interval_sets_initialization(self, empty_model):
        """
        Verify model handles empty interval sets correctly.
        """
        assert len(empty_model.rows) == 0, (
            f"Empty model should have 0 rows, found {len(empty_model.rows)}"
        )
        assert empty_model.rowCount() == 0, (
            f"Empty model rowCount() should return 0, found {empty_model.rowCount()}"
        )
        assert empty_model.columnCount() == EXPECTED_COLUMN_COUNT, (
            f"Empty model should still have {EXPECTED_COLUMN_COUNT} columns, "
            f"found {empty_model.columnCount()}"
        )

    def test_empty_model_header_data(self, empty_model):
        """
        Verify headers still work with empty model.
        """
        for col, expected in enumerate(EXPECTED_HEADERS):
            header = empty_model.headerData(
                col,
                Qt.Orientation.Horizontal,
                Qt.ItemDataRole.DisplayRole
            )
            assert header == expected, (
                f"Empty model, column {col}: Expected header '{expected}', found '{header}'"
            )

    def test_empty_model_invalid_index_handling(self, empty_model):
        """
        Verify empty model handles invalid indices gracefully.
        """
        index = empty_model.index(0, get_column_index("Interval Set"))
        assert not index.isValid(), (
            "Index (0,0) should be invalid in empty model"
        )

        # Test data() with invalid index
        data = empty_model.data(index, Qt.ItemDataRole.DisplayRole)
        assert data is None, "Invalid indexing should return None"

        # Test setData() with invalid index
        result = empty_model.setData(index, "value", Qt.ItemDataRole.EditRole)
        assert not result, "Invalid indexing setData should return False"

    def test_full_model_invalid_index_handling(self, model):
        """
        Verify model handles invalid indices gracefully.
        """
        # Invalid row
        index = model.index(999, get_column_index("Interval Set"))
        assert not index.isValid(), (
            "Index (999,0) should be invalid"
        )
        data = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert data is None, "Invalid row indexing should return None"
        result = model.setData(index, "value", Qt.ItemDataRole.EditRole)
        assert not result, "Invalid row indexing setData should return False"

        # Invalid column
        index = model.index(0, 999)
        assert not index.isValid(), (
            "Index (0,999) should be invalid"
        )
        data = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert data is None, "Invalid column indexing should return None"
        result = model.setData(index, "value", Qt.ItemDataRole.EditRole)
        assert not result, "Invalid column indexing setData should return False"


class TestSpinDelegate:
    """Test suite for SpinDelegate (alpha column 0.0-1.0)."""

    @pytest.fixture
    def delegate(self):
        """Create a SpinDelegate instance."""
        return SpinDelegate()

    @pytest.fixture
    def mock_parent(self, qtbot):
        """Create a mock parent widget."""
        from PySide6.QtWidgets import QWidget
        widget = QWidget()
        qtbot.addWidget(widget)
        return widget

    @pytest.fixture
    def mock_model(self, sample_interval_sets):
        """Create a model for testing."""
        return IntervalSetsModel(sample_interval_sets)

    # ========== TEST 7: Initialization ==========

    def test_initialization(self, delegate):
        """
        Verify delegate initializes correctly.

        SpinDelegate doesn't take min/max parameters (hardcoded 0.0-1.0).
        """
        assert delegate is not None


    def test_create_editor_returns_spinbox(self, delegate, mock_parent, mock_model):
        """
        Verify createEditor returns a QDoubleSpinBox.
        """
        index = mock_model.index(0, get_column_index("Alpha"))
        editor = delegate.createEditor(mock_parent, None, index)

        assert isinstance(editor, QDoubleSpinBox), (
            f"Expected QDoubleSpinBox, got {type(editor)}"
        )

    def test_create_editor_sets_range(self, delegate, mock_parent, mock_model):
        """
        Verify createEditor sets min/max to 0.0-1.0.
        """
        index = mock_model.index(0, get_column_index("Alpha"))
        editor = delegate.createEditor(mock_parent, None, index)

        assert editor.minimum() == 0.0, (
            f"Spinbox minimum should be 0.0, found {editor.minimum()}"
        )
        assert editor.maximum() == 1.0, (
            f"Spinbox maximum should be 1.0, found {editor.maximum()}"
        )

    def test_create_editor_sets_properties(self, delegate, mock_parent, mock_model):
        """
        Verify createEditor sets spinbox properties.
        """
        index = mock_model.index(0, get_column_index("Alpha"))
        editor = delegate.createEditor(mock_parent, None, index)

        assert editor.singleStep() == 0.1, "Single step should be 0.1"
        assert editor.decimals() == 1, "Decimals should be 1"

    def test_set_editor_data_with_valid_value(self, delegate, mock_parent, mock_model):
        """
        Verify setEditorData loads value from model into editor.
        """
        index = mock_model.index(0, get_column_index("Alpha"))  # alpha column
        expected_value = mock_model.rows[0]["alpha"]

        editor = delegate.createEditor(mock_parent, None, index)
        delegate.setEditorData(editor, index)

        assert editor.value() == expected_value, (
            f"Editor value {editor.value()} doesn't match model value {expected_value}"
        )

    def test_set_editor_data_with_none_value(self, delegate, mock_parent, mock_model):
        """
        Verify setEditorData handles None by setting 0.0.
        """
        index = mock_model.index(0, get_column_index("Alpha"))
        original_value = mock_model.rows[0]["alpha"]
        mock_model.rows[0]["alpha"] = None

        editor = delegate.createEditor(mock_parent, None, index)
        delegate.setEditorData(editor, index)

        assert editor.value() == 0.0, (
            "Editor should default to 0.0 when model value is None"
        )

        # Restore original value
        mock_model.rows[0]["alpha"] = original_value

    @pytest.mark.parametrize("test_value", [0.0, 0.5, 1.0, 0.3])
    def test_set_editor_data_various_values(self, delegate, mock_parent, mock_model, test_value):
        """
        Verify setEditorData with various numeric values.
        """
        index = mock_model.index(0, get_column_index("Alpha"))
        mock_model.rows[0]["alpha"] = test_value

        editor = delegate.createEditor(mock_parent, None, index)
        delegate.setEditorData(editor, index)

        assert editor.value() == test_value, (
            f"Editor value {editor.value()} doesn't match expected {test_value}"
        )

    def test_set_model_data(self, delegate, mock_parent, mock_model, qtbot):
        """
        Verify setModelData writes editor value back to model.
        """
        index = mock_model.index(0, get_column_index("Alpha"))
        test_value = 0.7

        editor = delegate.createEditor(mock_parent, None, index)
        editor.setValue(test_value)

        with qtbot.waitSignal(mock_model.dataChanged):
            delegate.setModelData(editor, mock_model, index)

        assert mock_model.rows[0]["alpha"] == test_value, (
            f"Model value {mock_model.rows[0]['alpha']} doesn't match "
            f"editor value {test_value}"
        )

    def test_set_model_data_calls_interpret_text(self, delegate, mock_parent, mock_model):
        """
        Verify setModelData calls interpretText before saving.

        This simulates user typing without pressing Enter.
        """
        index = mock_model.index(0, get_column_index("Alpha"))

        editor = delegate.createEditor(mock_parent, None, index)
        # Set text without programmatically setting value
        editor.lineEdit().setText("0.8")

        delegate.setModelData(editor, mock_model, index)

        # interpretText() should have parsed the text
        assert mock_model.rows[0]["alpha"] == 0.8, (
            "setModelData should call interpretText() to parse pending text"
        )


    def test_full_edit_cycle(self, delegate, mock_parent, mock_model, qtbot):
        """
        Test complete edit cycle: create -> load -> edit -> save.
        """
        index = mock_model.index(0, get_column_index("Alpha"))
        original_value = mock_model.rows[0]["alpha"]
        new_value = 0.9

        # Create editor
        editor = delegate.createEditor(mock_parent, None, index)
        assert isinstance(editor, QDoubleSpinBox)

        # Load data from model
        delegate.setEditorData(editor, index)
        assert editor.value() == original_value

        # User edits the value
        editor.setValue(new_value)
        assert editor.value() == new_value

        # Save data back to model
        with qtbot.waitSignal(mock_model.dataChanged):
            delegate.setModelData(editor, mock_model, index)

        assert mock_model.rows[0]["alpha"] == new_value, (
            f"Full edit cycle failed: expected {new_value}, "
            f"got {mock_model.rows[0]['alpha']}"
        )


class TestComboDelegate:
    """Test suite for ComboDelegate (color dropdown)."""

    @pytest.fixture
    def delegate(self):
        """Create a ComboDelegate instance."""
        return ComboDelegate()

    @pytest.fixture
    def mock_parent(self, qtbot):
        """Create a mock parent widget."""
        from PySide6.QtWidgets import QWidget
        widget = QWidget()
        qtbot.addWidget(widget)
        return widget

    @pytest.fixture
    def mock_model(self, sample_interval_sets):
        """Create a model for testing."""
        return IntervalSetsModel(sample_interval_sets)

    def test_create_editor_returns_combobox(self, delegate, mock_parent, mock_model):
        """
        Verify createEditor returns a QComboBox.
        """
        index = mock_model.index(0, get_column_index("Color"))
        editor = delegate.createEditor(mock_parent, None, index)

        assert isinstance(editor, QComboBox), (
            f"Expected QComboBox, got {type(editor)}"
        )

    def test_create_editor_has_color_items(self, delegate, mock_parent, mock_model):
        """
        Verify createEditor populates combobox with colors.
        """
        index = mock_model.index(0, get_column_index("Color"))
        editor = delegate.createEditor(mock_parent, None, index)

        # Check that all colors from GRADED_COLOR_LIST are present
        assert editor.count() == len(GRADED_COLOR_LIST), (
            f"Expected {len(GRADED_COLOR_LIST)} items, found {editor.count()}"
        )

        for i, expected_color in enumerate(GRADED_COLOR_LIST):
            assert editor.itemText(i) == expected_color, (
                f"Item {i}: Expected '{expected_color}', found '{editor.itemText(i)}'"
            )

    def test_set_editor_data_with_valid_color(self, delegate, mock_parent, mock_model):
        """
        Verify setEditorData sets correct color in combobox.
        """
        index = mock_model.index(0, get_column_index("Color"))  # color column
        expected_color = mock_model.rows[0]["colors"]

        editor = delegate.createEditor(mock_parent, None, index)
        delegate.setEditorData(editor, index)

        assert editor.currentText() == expected_color, (
            f"Editor shows '{editor.currentText()}', expected '{expected_color}'"
        )

    def test_set_editor_data_with_invalid_color(self, delegate, mock_parent, mock_model):
        """
        Verify setEditorData handles color not in list.

        If color not found, combobox should remain at index 0 or unchanged.
        """
        index = mock_model.index(0, get_column_index("Color"))
        mock_model.rows[0]["colors"] = "not_a_valid_color"

        editor = delegate.createEditor(mock_parent, None, index)
        delegate.setEditorData(editor, index)

        # Should stay at default (no match found, findText returns -1)
        # Implementation keeps current index unchanged if not found
        assert editor.currentIndex() >= 0, (
            "Combobox should have a valid index even with invalid color"
        )

    def test_set_model_data(self, delegate, mock_parent, mock_model, qtbot):
        """
        Verify setModelData writes combobox selection to model.
        """
        index = mock_model.index(0, get_column_index("Color"))
        test_color = GRADED_COLOR_LIST[2]  # Pick a color from the list

        editor = delegate.createEditor(mock_parent, None, index)
        editor.setCurrentText(test_color)

        with qtbot.waitSignal(mock_model.dataChanged):
            delegate.setModelData(editor, mock_model, index)

        assert mock_model.rows[0]["colors"] == test_color, (
            f"Model value '{mock_model.rows[0]['colors']}' doesn't match "
            f"editor value '{test_color}'"
        )


    def test_full_edit_cycle(self, delegate, mock_parent, mock_model, qtbot):
        """
        Test complete edit cycle for color selection.
        """
        index = mock_model.index(0, get_column_index("Color"))
        original_color = mock_model.rows[0]["colors"]
        new_color = GRADED_COLOR_LIST[-1]  # Pick last color

        # Create editor
        editor = delegate.createEditor(mock_parent, None, index)
        assert isinstance(editor, QComboBox)

        # Load data from model
        delegate.setEditorData(editor, index)
        assert editor.currentText() == original_color

        # User selects new color
        editor.setCurrentText(new_color)
        assert editor.currentText() == new_color

        # Save data back to model
        with qtbot.waitSignal(mock_model.dataChanged):
            delegate.setModelData(editor, mock_model, index)

        assert mock_model.rows[0]["colors"] == new_color, (
            f"Full edit cycle failed: expected '{new_color}', "
            f"got '{mock_model.rows[0]['colors']}'"
        )

class TestIntervalSetsDialog:
    """Test suite for IntervalSetsDialog."""

    @pytest.fixture
    def model(self, sample_interval_sets):
        """Create an IntervalSetsModel instance."""
        return IntervalSetsModel(sample_interval_sets)

    @pytest.fixture
    def dialog(self, model, qtbot):
        """Create an IntervalSetsDialog instance."""
        dialog = IntervalSetsDialog(model)
        qtbot.addWidget(dialog)
        return dialog

    def test_dialog_initialization(self, dialog, model):
        """
        Verify dialog initializes correctly.
        """
        assert dialog.view is not None, "View should be initialized"
        assert dialog.view.model() is model, "View should use the provided model"

    def test_window_properties(self, dialog):
        """
        Verify dialog has correct window properties.
        """
        assert dialog.windowTitle() == "Interval Sets"
        assert dialog.minimumSize().width() == 400
        assert dialog.minimumSize().height() == 300

    def test_delegates_assigned_correctly(self, dialog):
        """
        Verify correct delegates are assigned to columns based on header names.

        Uses EXPECTED_HEADERS to map column names to indices:
        - "Color" -> ComboDelegate
        - "Alpha" -> SpinDelegate
        - "Interval Set" -> No custom delegate (default for checkbox)
        """
        # Use EXPECTED_HEADERS to get column indices by name
        color_col = EXPECTED_HEADERS.index("Color")
        alpha_col = EXPECTED_HEADERS.index("Alpha")
        interval_col = EXPECTED_HEADERS.index("Interval Set")

        # Verify Color column has ComboDelegate
        color_delegate = dialog.view.itemDelegateForColumn(color_col)
        assert isinstance(color_delegate, ComboDelegate), (
            f"'Color' column (index {color_col}) should have ComboDelegate, "
            f"found {type(color_delegate)}"
        )

        # Verify Alpha column has SpinDelegate
        alpha_delegate = dialog.view.itemDelegateForColumn(alpha_col)
        assert isinstance(alpha_delegate, SpinDelegate), (
            f"'Alpha' column (index {alpha_col}) should have SpinDelegate, "
            f"found {type(alpha_delegate)}"
        )

        # Verify Interval Set column has no custom delegate
        interval_delegate = dialog.view.itemDelegateForColumn(interval_col)
        assert not isinstance(interval_delegate, (ComboDelegate, SpinDelegate)), (
            f"'Interval Set' column (index {interval_col}) should use default delegate, "
            f"found {type(interval_delegate)}"
        )

    def test_color_editor_creation(self, dialog, model, qtbot):
        """
        Verify color delegate creates correct editor.
        """
        delegate = dialog.view.itemDelegateForColumn(get_column_index("Color"))
        index = model.index(0, get_column_index("Color"))

        editor = delegate.createEditor(dialog.view, None, index)
        qtbot.addWidget(editor)

        assert isinstance(editor, QComboBox), (
            f"Color editor should be QComboBox, found {type(editor)}"
        )
        assert editor.count() == len(GRADED_COLOR_LIST)

    def test_alpha_editor_creation(self, dialog, model, qtbot):
        """
        Verify alpha delegate creates correct editor.
        """
        delegate = dialog.view.itemDelegateForColumn(get_column_index("Alpha"))
        index = model.index(0, get_column_index("Alpha"))

        editor = delegate.createEditor(dialog.view, None, index)
        qtbot.addWidget(editor)

        assert isinstance(editor, QDoubleSpinBox), (
            f"Alpha editor should be QDoubleSpinBox, found {type(editor)}"
        )
        assert editor.minimum() == 0.0
        assert editor.maximum() == 1.0

    def test_color_edit_updates_model(self, dialog, model, qtbot):
        """
        Verify editing color updates the model correctly.
        """
        delegate = dialog.view.itemDelegateForColumn(get_column_index("Color"))
        index = model.index(0, get_column_index("Color"))
        original_color = model.rows[0]["colors"]
        new_color = GRADED_COLOR_LIST[-1]

        # Create and populate editor
        editor = delegate.createEditor(dialog.view, None, index)
        qtbot.addWidget(editor)
        delegate.setEditorData(editor, index)

        # Verify initial value
        assert editor.currentText() == original_color

        # Change value
        editor.setCurrentText(new_color)

        # Save back to model
        with qtbot.waitSignal(model.dataChanged):
            delegate.setModelData(editor, model, index)

        # Verify model was updated
        assert model.rows[0]["colors"] == new_color, (
            f"Model should have new color '{new_color}', "
            f"found '{model.rows[0]['colors']}'"
        )

    def test_alpha_edit_updates_model(self, dialog, model, qtbot):
        """
        Verify editing alpha updates the model correctly.
        """
        delegate = dialog.view.itemDelegateForColumn(get_column_index("Alpha"))
        index = model.index(0, get_column_index("Alpha"))
        original_alpha = model.rows[0]["alpha"]
        new_alpha = 0.8

        # Create and populate editor
        editor = delegate.createEditor(dialog.view, None, index)
        qtbot.addWidget(editor)
        delegate.setEditorData(editor, index)

        # Verify initial value
        assert editor.value() == original_alpha

        # Change value
        editor.setValue(new_alpha)

        # Save back to model
        with qtbot.waitSignal(model.dataChanged):
            delegate.setModelData(editor, model, index)

        # Verify model was updated
        assert model.rows[0]["alpha"] == new_alpha, (
            f"Model should have new alpha {new_alpha}, "
            f"found {model.rows[0]['alpha']}"
        )

    def test_ok_button_accepts_dialog(self, dialog, qtbot):
        """
        Verify OK button accepts the dialog.
        """
        ok_button = None
        for button in dialog.findChildren(QPushButton):
            if button.text() == "OK":
                ok_button = button
                break

        assert ok_button is not None, "OK button not found"

        with qtbot.waitSignal(dialog.accepted):
            ok_button.click()

    def test_cancel_button_rejects_dialog(self, dialog, qtbot):
        """
        Verify Cancel button rejects the dialog.
        """
        cancel_button = None
        for button in dialog.findChildren(QPushButton):
            if button.text() == "Cancel":
                cancel_button = button
                break

        assert cancel_button is not None, "Cancel button not found"

        with qtbot.waitSignal(dialog.rejected):
            cancel_button.click()
