
from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from pynaviz.utils import GRADED_COLOR_LIST


class IntervalSetsModel(QAbstractTableModel):
    """A model to handle the dict of interval sets with checkboxes."""

    checkStateChanged = pyqtSignal(str, str, float, bool)

    def __init__(self, interval_sets: dict):
        super().__init__()
        self.interval_sets = interval_sets
        self.colors = GRADED_COLOR_LIST
        self.rows = [
            {
                "name": k,
                "colors": self.colors[i%len(self.colors)],
                "alpha": 0.5,
                "checked": False
            }
            for i, k in enumerate(interval_sets.keys())
        ]

    # ---- model dimensions ----
    def rowCount(self, parent=None):
        if parent is None:
            parent = QModelIndex()
        return len(self.rows)

    def columnCount(self, parent=None):
        return 3

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return ["Interval Set", "Color", "Alpha"][section]

    def data(self, index, role=None):
        """What to display in the table view."""
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        r = self.rows[row]

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if col == 0:
                return r["name"]
            if col == 1:
                return r["colors"]
            if col == 2:
                return r["alpha"]

        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            return Qt.CheckState.Checked if r["checked"] else Qt.CheckState.Unchecked

        return None

    def flags(self, index):
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            return base | Qt.ItemFlag.ItemIsUserCheckable
        elif index.column() == 1:
            return base | Qt.ItemFlag.ItemIsEditable
        elif index.column() == 2:
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index, value, role=None):
        """
        Write data to the model.

        Parameters
        ----------
        index : QModelIndex
            The index of the item to modify.
        value : Any
            The new value to set.
        role : Qt.ItemDataRole
            The role of the data to set.
        """
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        r = self.rows[row]

        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            check_value = getattr(value, 'value', value)
            r["checked"] = (check_value == Qt.CheckState.Checked.value)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            self.checkStateChanged.emit(r["name"], r["colors"], r["alpha"], r["checked"])
            return True

        if role == Qt.ItemDataRole.EditRole:
            if col == 1:
                r["colors"] = str(value)
            elif col == 2:
                r["alpha"] = float(value)
            else:
                return False
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.EditRole])
            self.checkStateChanged.emit(r["name"], r["colors"], r["alpha"], r["checked"])
            return True
        return False


class ComboDelegate(QStyledItemDelegate):
    """Drop-down editor for colors."""
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(GRADED_COLOR_LIST)
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        if value:
            i = editor.findText(value)
            if i >= 0:
                editor.setCurrentIndex(i)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), role=Qt.ItemDataRole.EditRole)


class SpinDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        spin = QDoubleSpinBox(parent)
        spin.setMinimum(0.0)
        spin.setMaximum(1.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(1)
        return spin

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        if value is None:
            value = 0.0
        editor.setValue(float(value))

    def setModelData(self, editor, model, index):
        editor.interpretText()  # ensure the text is parsed
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)


class IntervalSetsDialog(QDialog):
    """
    Dialog showing a table of interval sets with:
    - Column 0: name + checkbox
    - Column 1: dropdown
    - Column 2: number entry
    """
    def __init__(self, model: IntervalSetsModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Interval Sets")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumSize(400, 300)

        self.view = QTableView(self)
        self.view.setModel(model)
        header = self.view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)


        color_delegate = ComboDelegate(self.view)
        self.view.setItemDelegateForColumn(1, color_delegate)

        alpha_delegate = SpinDelegate(self.view)
        self.view.setItemDelegateForColumn(2, alpha_delegate)

        layout = QVBoxLayout()
        layout.addWidget(self.view)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.setDefault(True)
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.adjustSize()
