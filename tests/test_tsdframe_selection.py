from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDoubleSpinBox, QPushButton

from pynaviz.qt.tsdframe_selection import (
    GRADED_COLOR_LIST,
    ComboDelegate,
    DoubleSpinDelegate,
    TsdFramesDialog,
    TsdFramesModel,
)

# CURRENT FRAME MODEL CONFIGS - Ideally Making Refactoring Easier
COLUMNS_IDS = OrderedDict({
    "name": 0,
    "colors": 1,
    "markersize": 2,
    "thickness": 3,
})
KEYS = set(COLUMNS_IDS.keys())
KEYS.add("checked")
KEY_TYPES = {
    "name": str,
    "colors": str,
    "markersize": (int, float),
    "thickness": (int, float),
    "checked": bool,
}
KEY_DEFAULTS = {
    "name": None,
    "colors": None,
    "markersize": 10,
    "thickness": 2,
    "checked": False,
}
EXPECTED_HEADERS = ["TsdFrame", "Color", "Size", "Thickness"]
EXPECTED_COLUMN_COUNT = len(COLUMNS_IDS)
EXPECTED_ROW_KEYS_COUNT = len(KEYS)

# Flag configurations for each column
COLUMN_FLAGS = {
    0: Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable,
    1: Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable,
    2: Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable,
    3: Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable,
}

# Role configurations
SUPPORTED_DATA_ROLES = {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}
SUPPORTED_HEADER_ROLES = {Qt.ItemDataRole.DisplayRole}
CHECKSTATE_COLUMN = 0  # Only column 0 supports CheckStateRole


@pytest.fixture
def sample_tsdframes():
    """Create sample tsdframes dict for testing."""
    return {
        "frame1": MagicMock(),
        "frame2": MagicMock(),
        "frame3": MagicMock(),
    }


class TestTsdFramesModel:
    """Test suite for TsdFramesModel."""


    @pytest.fixture
    def model(self, sample_tsdframes):
        """Create a TsdFramesModel instance."""
        return TsdFramesModel(sample_tsdframes)

    @pytest.fixture
    def empty_model(self):
        """Create a TsdFramesModel instance with empty data."""
        return TsdFramesModel({})

    @pytest.fixture
    def all_item_data_roles(self):
        """Get all ItemDataRole enum values."""
        return [
            getattr(Qt.ItemDataRole, r)
            for r in dir(Qt.ItemDataRole)
            if r.endswith("Role")
        ]

    # ========== Test Initialization ==========

    def test_initialization_rows_count(self, model, sample_tsdframes):
        """Test that model creates correct number of rows."""
        assert len(model.rows) == len(sample_tsdframes), (
            f"Expected {len(sample_tsdframes)} rows, found {len(model.rows)}"
        )
        assert model.rowCount() == len(sample_tsdframes), (
            f"rowCount() returned {model.rowCount()}, expected {len(sample_tsdframes)}"
        )

    def test_initialization_row_structure(self, model, sample_tsdframes):
        """Test that each row has the correct structure and data types."""
        for i, (key_tsdframe, row) in enumerate(zip(sample_tsdframes.keys(), model.rows)):
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
            for key, default in KEY_DEFAULTS.items():
                if default is not None:
                    assert row[key] == default, (
                        f"Row {i}, key '{key}': Expected default {default}, found {row[key]}"
                    )

            # Check name matches tsdframe key
            assert row["name"] == key_tsdframe, (
                f"Row {i}: Expected name '{key_tsdframe}', found '{row['name']}'"
            )

            # Check color assignment
            assert row["colors"] == GRADED_COLOR_LIST[i % len(GRADED_COLOR_LIST)], (
                f"Row {i}: Color mismatch. Expected {GRADED_COLOR_LIST[i % len(GRADED_COLOR_LIST)]}, "
                f"found {row['colors']}"
            )

    def test_row_count_column_consistency(self, model):
        """Test that row count matches number of entries in rows list."""
        assert model.rowCount() == len(model.rows), (
            f"rowCount() {model.rowCount()} doesn't match len(rows) {len(model.rows)}"
        )
        assert model.columnCount() == EXPECTED_COLUMN_COUNT, (
            f"Expected {EXPECTED_COLUMN_COUNT} columns, found {model.columnCount()}"
        )

    # ========== Test Header Data ==========

    def test_header_data(self, model):
        """Test that header labels are correct for all columns."""
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
        """Test that header returns None for unsupported roles."""
        unsupported_roles = [
            role for role in all_item_data_roles
            if role not in SUPPORTED_HEADER_ROLES
        ]

        for col in range(EXPECTED_COLUMN_COUNT):
            for role in unsupported_roles:
                result = model.headerData(col, Qt.Orientation.Horizontal, role)
                assert result is None, (
                    f"Column {col}, Role {role}: Expected None, found {result}"
                )

    def test_header_data_vertical_orientation(self, model, all_item_data_roles):
        """Test that vertical headers return None (not implemented)."""
        for col in range(EXPECTED_COLUMN_COUNT):
            for role in all_item_data_roles:
                result = model.headerData(col, Qt.Orientation.Vertical, role)
                assert result is None, (
                    f"Vertical header for column {col}, role {role}: "
                    f"Expected None, found {result}"
                )

    # ========== Test Data Method ==========

    def test_data_display_role_by_columns(self, model):
        """Test data() returns correct values for DisplayRole across all columns and rows."""
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
        """Test that EditRole returns same as DisplayRole for all cells."""
        for row in range(model.rowCount()):
            for col in range(model.columnCount()):
                index = model.index(row, col)
                display = model.data(index, Qt.ItemDataRole.DisplayRole)
                edit = model.data(index, Qt.ItemDataRole.EditRole)
                assert display == edit, (
                    f"Row {row}, Column {col}: DisplayRole ({display}) != EditRole ({edit})"
                )

    def test_data_checkstate_role_column_0_all_rows(self, model):
        """Test CheckStateRole for column 0 across all rows."""
        for row in range(model.rowCount()):
            index = model.index(row, 0)

            # Test unchecked state
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
        """Test that CheckStateRole returns None for columns 1-3 across all rows."""
        for row in range(model.rowCount()):
            for col in range(1, EXPECTED_COLUMN_COUNT):
                index = model.index(row, col)
                checkstate = model.data(index, Qt.ItemDataRole.CheckStateRole)
                assert checkstate is None, (
                    f"Row {row}, Column {col}: CheckStateRole should return None, found {checkstate}"
                )

    def test_data_unsupported_roles(self, model, all_item_data_roles):
        """Test that data() returns None for unsupported roles."""
        unsupported_roles = [
            role for role in all_item_data_roles
            if role not in SUPPORTED_DATA_ROLES and role != Qt.ItemDataRole.CheckStateRole
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

    # ========== Test Flags ==========

    def test_flags_all_columns(self, model):
        """Test flags for all columns across all rows."""
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
        """Test that base flags (Enabled, Selectable) are always present."""
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

        for row in range(model.rowCount()):
            for col in range(model.columnCount()):
                index = model.index(row, col)
                flags = model.flags(index)
                assert (flags & base) == base, (
                    f"Row {row}, Column {col}: Base flags missing. Found flags: {flags}"
                )

    # ========== Test SetData ==========

    @pytest.mark.parametrize("check_state,expected_bool", [
        (Qt.CheckState.Checked, True),
        (Qt.CheckState.Unchecked, False),
    ])
    def test_setdata_checkstate(self, model, qtbot, check_state, expected_bool):
        """Test setting check state for all rows."""
        for row in range(model.rowCount()):
            index = model.index(row, CHECKSTATE_COLUMN)

            with qtbot.waitSignal(model.checkStateChanged) as blocker:
                result = model.setData(index, check_state, Qt.ItemDataRole.CheckStateRole)

            assert result is True, (
                f"Row {row}: setData returned False for CheckStateRole"
            )
            assert model.rows[row]["checked"] is expected_bool, (
                f"Row {row}: Expected checked={expected_bool}, found {model.rows[row]['checked']}"
            )
            assert blocker.args[0] == model.rows[row]["name"], (
                f"Row {row}: Signal arg[0] (name) mismatch"
            )
            assert blocker.args[4] is expected_bool, (
                f"Row {row}: Signal arg[4] (checked) expected {expected_bool}, found {blocker.args[4]}"
            )

    @pytest.mark.parametrize("col,key,test_values", [
        (1, "colors", ["red", "blue", "#FF5733"]),
        (2, "markersize", [0.0, 15.5, 100.0]),
        (3, "thickness", [0.5, 3.7, 10.0]),
    ])
    def test_setdata_edit_role(self, model, qtbot, col, key, test_values):
        """Test editing values in editable columns across all rows."""
        signal_arg_index = col  # Signal args match column index for cols 1-3

        for row in range(model.rowCount()):
            for test_value in test_values:
                index = model.index(row, col)

                with qtbot.waitSignal(model.checkStateChanged) as blocker:
                    result = model.setData(index, test_value, Qt.ItemDataRole.EditRole)

                assert result is True, (
                    f"Row {row}, Column {col} ('{key}'): "
                    f"setData returned False for value {test_value}"
                )
                assert model.rows[row][key] == test_value, (
                    f"Row {row}, Column {col} ('{key}'): "
                    f"Expected {test_value}, found {model.rows[row][key]}"
                )
                assert blocker.args[signal_arg_index] == test_value, (
                    f"Row {row}, Column {col} ('{key}'): "
                    f"Signal arg[{signal_arg_index}] expected {test_value}, "
                    f"found {blocker.args[signal_arg_index]}"
                )

    def test_setdata_edit_role_column_0_fails(self, model):
        """Test that editing column 0 with EditRole fails for all rows."""
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
        """Test that setData returns False for unsupported roles."""
        unsupported_roles = [
            role for role in all_item_data_roles
            if role not in {Qt.ItemDataRole.EditRole, Qt.ItemDataRole.CheckStateRole}
        ]

        for row in range(min(2, model.rowCount())):  # Test first 2 rows for efficiency
            for col in range(model.columnCount()):
                for role in unsupported_roles:
                    index = model.index(row, col)
                    result = model.setData(index, "test_value", role)
                    assert result is False, (
                        f"Row {row}, Column {col}, Role {role}: "
                        f"setData should return False for unsupported role"
                    )

    def test_setdata_emits_datachanged(self, model, qtbot):
        """Test that setData emits dataChanged signal with correct parameters."""
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

    # ========== Integration Tests ==========

    def test_multiple_rows_independence(self, model):
        """Test that modifying one row doesn't affect others."""
        if model.rowCount() < 2:
            pytest.skip("Need at least 2 rows for this test")

        # Store original values for all rows except first
        original_values = [
            {
                "checked": model.rows[row]["checked"],
                "colors": model.rows[row]["colors"],
                "markersize": model.rows[row]["markersize"],
                "thickness": model.rows[row]["thickness"],
            }
            for row in range(1, model.rowCount())
        ]

        # Modify first row
        model.setData(model.index(0, 0), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
        model.setData(model.index(0, 1), "modified_color", Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 2), 99.9, Qt.ItemDataRole.EditRole)
        model.setData(model.index(0, 3), 88.8, Qt.ItemDataRole.EditRole)

        # Check other rows are unaffected
        for i, row in enumerate(range(1, model.rowCount())):
            for key, original_value in original_values[i].items():
                current_value = model.rows[row][key]
                assert current_value == original_value, (
                    f"Row {row}, key '{key}': Value changed unexpectedly. "
                    f"Expected {original_value}, found {current_value}"
                )

    # ========== Test Empty TsdFrames ==========

    def test_empty_tsdframes_initialization(self, empty_model):
        """Test that model handles empty tsdframes correctly."""
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
        """Test that headers still work with empty model."""
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
        """Test that empty model handles invalid indices gracefully."""
        index = empty_model.index(0, 0)
        assert not index.isValid(), (
            "Index (0,0) should be invalid in empty model"
        )

        # Test data() with invalid index
        data = empty_model.data(index, Qt.ItemDataRole.DisplayRole)
        assert data is None, "Invalid indexing returned a data value."
        # Implementation dependent - might return None or handle gracefully

        # Test setData() with invalid index
        result = empty_model.setData(index, "value", Qt.ItemDataRole.EditRole)
        # Should return False or handle gracefully without crashing
        assert not result, "Invalid indexing returned a True."

    def test_full_model_invalid_index_handling(self, model):
        """Test that empty model handles invalid indices gracefully."""
        # invalid col
        index = model.index(4, 0)
        assert not index.isValid(), (
            "Index (4,0) should be invalid in 3 rows model"
        )

        # Test data() with invalid index
        data = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert data is None, "Invalid row indexing returned a data value."
        # Implementation dependent - might return None or handle gracefully

        # Test setData() with invalid index
        result = model.setData(index, "value", Qt.ItemDataRole.EditRole)
        # Should return False or handle gracefully without crashing
        assert not result, "Invalid row indexing returned a True."

        # invalid row
        index = model.index(0, 5)
        assert not index.isValid(), (
            "Index (0,5) should be invalid in 4 cols model"
        )

        # Test data() with invalid index
        data = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert data is None, "Invalid row indexing returned a data value."
        # Implementation dependent - might return None or handle gracefully

        # Test setData() with invalid index
        result = model.setData(index, "value", Qt.ItemDataRole.EditRole)
        # Should return False or handle gracefully without crashing
        assert not result, "Invalid row indexing returned a True."


class TestDoubleSpinDelegate:
    """Test suite for DoubleSpinDelegate."""

    @pytest.fixture
    def delegate(self):
        """Create a DoubleSpinDelegate instance."""
        return DoubleSpinDelegate(min_=0.0, max_=100.0)

    @pytest.fixture
    def mock_parent(self, qtbot):
        """Create a mock parent widget."""
        from PyQt6.QtWidgets import QWidget
        widget = QWidget()
        qtbot.addWidget(widget)
        return widget

    @pytest.fixture
    def mock_model(self, sample_tsdframes):
        """Create a model for testing."""
        return TsdFramesModel(sample_tsdframes)

    # ========== Test Initialization ==========

    def test_initialization(self, delegate):
        """Test that delegate initializes with correct min/max."""
        assert delegate.min_ == 0.0, "Min value not set correctly"
        assert delegate.max_ == 100.0, "Max value not set correctly"

    # ========== Test createEditor ==========

    def test_create_editor_returns_spinbox(self, delegate, mock_parent, mock_model):
        """Test that createEditor returns a QDoubleSpinBox."""
        index = mock_model.index(0, 2)
        editor = delegate.createEditor(mock_parent, None, index)

        assert isinstance(editor, QDoubleSpinBox), (
            f"Expected QDoubleSpinBox, got {type(editor)}"
        )

    def test_create_editor_sets_range(self, delegate, mock_parent, mock_model):
        """Test that createEditor sets min/max on the spinbox."""
        index = mock_model.index(0, 2)
        editor = delegate.createEditor(mock_parent, None, index)

        assert editor.minimum() == delegate.min_, (
            f"Spinbox minimum {editor.minimum()} doesn't match delegate min {delegate.min_}"
        )
        assert editor.maximum() == delegate.max_, (
            f"Spinbox maximum {editor.maximum()} doesn't match delegate max {delegate.max_}"
        )

    def test_create_editor_sets_properties(self, delegate, mock_parent, mock_model):
        """Test that createEditor sets spinbox properties correctly."""
        index = mock_model.index(0, 2)
        editor = delegate.createEditor(mock_parent, None, index)

        assert editor.singleStep() == 1.0, "Single step should be 1.0"
        assert editor.decimals() == 2, "Decimals should be 2"

    # ========== Test setEditorData ==========

    def test_set_editor_data_with_valid_value(self, delegate, mock_parent, mock_model):
        """Test that setEditorData loads value from model into editor."""
        index = mock_model.index(0, 2)  # markersize column
        expected_value = mock_model.rows[0]["markersize"]

        editor = delegate.createEditor(mock_parent, None, index)
        delegate.setEditorData(editor, index)

        assert editor.value() == expected_value, (
            f"Editor value {editor.value()} doesn't match model value {expected_value}"
        )

    def test_set_editor_data_with_none_value(self, delegate, mock_parent, mock_model):
        """Test that setEditorData handles None by setting 0.0."""
        index = mock_model.index(0, 2)
        # Temporarily set value to None
        original_value = mock_model.rows[0]["markersize"]
        mock_model.rows[0]["markersize"] = None

        editor = delegate.createEditor(mock_parent, None, index)
        delegate.setEditorData(editor, index)

        assert editor.value() == 0.0, (
            "Editor should default to 0.0 when model value is None"
        )

        # Restore original value
        mock_model.rows[0]["markersize"] = original_value

    @pytest.mark.parametrize("test_value", [5.5, 0.0, 100.0, 42.37])
    def test_set_editor_data_various_values(self, delegate, mock_parent, mock_model, test_value):
        """Test setEditorData with various numeric values."""
        index = mock_model.index(0, 2)
        mock_model.rows[0]["markersize"] = test_value

        editor = delegate.createEditor(mock_parent, None, index)
        delegate.setEditorData(editor, index)

        assert editor.value() == test_value, (
            f"Editor value {editor.value()} doesn't match expected {test_value}"
        )

    # ========== Test setModelData ==========

    def test_set_model_data(self, delegate, mock_parent, mock_model, qtbot):
        """Test that setModelData writes editor value back to model."""
        index = mock_model.index(0, 2)
        test_value = 25.75

        editor = delegate.createEditor(mock_parent, None, index)
        editor.setValue(test_value)

        with qtbot.waitSignal(mock_model.dataChanged):
            delegate.setModelData(editor, mock_model, index)

        assert mock_model.rows[0]["markersize"] == test_value, (
            f"Model value {mock_model.rows[0]['markersize']} doesn't match "
            f"editor value {test_value}"
        )

    def test_set_model_data_calls_interpret_text(self, delegate, mock_parent, mock_model):
        """Test that setModelData calls interpretText before saving."""
        index = mock_model.index(0, 2)

        editor = delegate.createEditor(mock_parent, None, index)
        # Set text without programmatically setting value
        # This simulates user typing "15.5" but not pressing Enter
        editor.lineEdit().setText("15.5")

        delegate.setModelData(editor, mock_model, index)

        # interpretText() should have parsed the text
        assert mock_model.rows[0]["markersize"] == 15.5, (
            "setModelData should call interpretText() to parse pending text"
        )

    # ========== Integration Test ==========

    def test_full_edit_cycle(self, delegate, mock_parent, mock_model, qtbot):
        """Test complete edit cycle: create -> load -> edit -> save."""
        index = mock_model.index(0, 2)
        original_value = mock_model.rows[0]["markersize"]
        new_value = 99.99

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

        assert mock_model.rows[0]["markersize"] == new_value, (
            f"Full edit cycle failed: expected {new_value}, "
            f"got {mock_model.rows[0]['markersize']}"
        )

    # ========== Edge Cases ==========

    @pytest.mark.parametrize("min_val,max_val", [
        (0, 100),
        (-100, 100),
        (0, 1e12),
        (-1e12, 1e12),
    ])
    def test_various_ranges(self, mock_parent, mock_model, min_val, max_val):
        """Test delegate with various min/max ranges."""
        delegate = DoubleSpinDelegate(min_=min_val, max_=max_val)
        index = mock_model.index(0, 2)

        editor = delegate.createEditor(mock_parent, None, index)

        assert editor.minimum() == min_val
        assert editor.maximum() == max_val


class TestTsdFramesDialog:
    """Test suite for TsdFramesDialog."""

    @pytest.fixture
    def sample_tsdframes(self):
        """Create sample tsdframes dict for testing."""
        return {
            "frame1": MagicMock(),
            "frame2": MagicMock(),
        }

    @pytest.fixture
    def model(self, sample_tsdframes):
        """Create a TsdFramesModel instance."""
        return TsdFramesModel(sample_tsdframes)

    @pytest.fixture
    def dialog(self, model, qtbot):
        """Create a TsdFramesDialog instance."""
        dialog = TsdFramesDialog(model)
        qtbot.addWidget(dialog)
        return dialog

    # ========== Test Initialization ==========

    def test_dialog_initialization(self, dialog, model):
        """Test that dialog initializes correctly."""
        assert dialog.view is not None, "View should be initialized"
        assert dialog.view.model() is model, "View should use the provided model"

    def test_window_properties(self, dialog):
        """Test that dialog has correct window properties."""
        assert dialog.windowTitle() == "TsdFrame selection"
        assert dialog.minimumSize().width() == 400
        assert dialog.minimumSize().height() == 300

    # ========== Test Delegates ==========

    def test_combo_delegate_column_1(self, dialog):
        """Test that ComboDelegate is set for column 1 (colors)."""
        delegate = dialog.view.itemDelegateForColumn(1)
        assert isinstance(delegate, ComboDelegate), (
            f"Column 1 should have ComboDelegate, found {type(delegate)}"
        )

    def test_markersize_delegate_column_2(self, dialog):
        """Test that DoubleSpinDelegate is set for column 2 (markersize)."""
        delegate = dialog.view.itemDelegateForColumn(2)
        assert isinstance(delegate, DoubleSpinDelegate), (
            f"Column 2 should have DoubleSpinDelegate, found {type(delegate)}"
        )
        assert delegate.min_ == 0, f"Markersize min should be 0, found {delegate.min_}"
        assert delegate.max_ == 1e12, f"Markersize max should be 1e12, found {delegate.max_}"

    def test_thickness_delegate_column_3(self, dialog):
        """Test that DoubleSpinDelegate is set for column 3 (thickness)."""
        delegate = dialog.view.itemDelegateForColumn(3)
        assert isinstance(delegate, DoubleSpinDelegate), (
            f"Column 3 should have DoubleSpinDelegate, found {type(delegate)}"
        )
        assert delegate.min_ == 0, f"Thickness min should be 0, found {delegate.min_}"
        assert delegate.max_ == 1e12, f"Thickness max should be 1e12, found {delegate.max_}"

    def test_column_0_has_no_custom_delegate(self, dialog):
        """Test that column 0 uses default delegate (for checkbox)."""
        delegate = dialog.view.itemDelegateForColumn(0)
        # Should be None or default delegate, not a custom one
        assert not isinstance(delegate, (ComboDelegate, DoubleSpinDelegate)), (
            f"Column 0 should use default delegate, found {type(delegate)}"
        )

    # ========== Test Delegate Functionality ==========

    def test_markersize_editor_creation(self, dialog, model, qtbot):
        """Test that markersize delegate creates correct editor."""
        delegate = dialog.view.itemDelegateForColumn(2)
        index = model.index(0, 2)

        editor = delegate.createEditor(dialog.view, None, index)
        qtbot.addWidget(editor)

        assert isinstance(editor, QDoubleSpinBox), (
            f"Markersize editor should be QDoubleSpinBox, found {type(editor)}"
        )
        assert editor.minimum() == 0
        assert editor.maximum() == 1e12

    def test_thickness_editor_creation(self, dialog, model, qtbot):
        """Test that thickness delegate creates correct editor."""
        delegate = dialog.view.itemDelegateForColumn(3)
        index = model.index(0, 3)

        editor = delegate.createEditor(dialog.view, None, index)
        qtbot.addWidget(editor)

        assert isinstance(editor, QDoubleSpinBox), (
            f"Thickness editor should be QDoubleSpinBox, found {type(editor)}"
        )
        assert editor.minimum() == 0
        assert editor.maximum() == 1e12

    def test_markersize_edit_updates_model(self, dialog, model, qtbot):
        """Test that editing markersize updates the model correctly."""
        delegate = dialog.view.itemDelegateForColumn(2)
        index = model.index(0, 2)
        original_value = model.rows[0]["markersize"]
        new_value = 42.5

        # Create and populate editor
        editor = delegate.createEditor(dialog.view, None, index)
        qtbot.addWidget(editor)
        delegate.setEditorData(editor, index)

        # Verify initial value
        assert editor.value() == original_value

        # Change value
        editor.setValue(new_value)

        # Save back to model
        with qtbot.waitSignal(model.dataChanged):
            delegate.setModelData(editor, model, index)

        # Verify model was updated
        assert model.rows[0]["markersize"] == new_value, (
            f"Model should have new markersize {new_value}, "
            f"found {model.rows[0]['markersize']}"
        )

    def test_thickness_edit_updates_model(self, dialog, model, qtbot):
        """Test that editing thickness updates the model correctly.

        This test mimicks Qt calls to setValue->setModelData and
        """
        delegate = dialog.view.itemDelegateForColumn(3)
        index = model.index(0, 3)
        original_value = model.rows[0]["thickness"]
        new_value = 5.75

        # Create and populate editor
        editor = delegate.createEditor(dialog.view, None, index)
        qtbot.addWidget(editor)
        delegate.setEditorData(editor, index)

        # Verify initial value
        assert editor.value() == original_value

        # Change value
        editor.setValue(new_value)

        # Save back to model
        with qtbot.waitSignal(model.dataChanged):
            delegate.setModelData(editor, model, index)

        # Verify model was updated
        assert model.rows[0]["thickness"] == new_value, (
            f"Model should have new thickness {new_value}, "
            f"found {model.rows[0]['thickness']}"
        )

    # ========== Test Dialog Buttons ==========

    def test_ok_button_accepts_dialog(self, dialog, qtbot):
        """Test that OK button accepts the dialog."""
        ok_button = None
        for button in dialog.findChildren(QPushButton):
            if button.text() == "OK":
                ok_button = button
                break

        assert ok_button is not None, "OK button not found"

        with qtbot.waitSignal(dialog.accepted):
            ok_button.click()

    def test_cancel_button_rejects_dialog(self, dialog, qtbot):
        """Test that Cancel button rejects the dialog."""
        cancel_button = None
        for button in dialog.findChildren(QPushButton):
            if button.text() == "Cancel":
                cancel_button = button
                break

        assert cancel_button is not None, "Cancel button not found"

        with qtbot.waitSignal(dialog.rejected):
            cancel_button.click()
