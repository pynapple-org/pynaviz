
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from pynaviz.qt.interval_sets_selection import ComboDelegate
from pynaviz.utils import GRADED_COLOR_LIST


class DoubleSpinDelegate(QStyledItemDelegate):

    def __init__(self, min_, max_, parent=None):
        super().__init__(parent)
        self.min_ = min_
        self.max_ = max_

    def createEditor(self, parent, option, index):
        spin = QDoubleSpinBox(parent)
        # Very wide range to simulate "no boundaries"
        spin.setMinimum(self.min_)
        spin.setMaximum(self.max_)
        spin.setSingleStep(1)
        spin.setDecimals(2)  # adjust precision as needed

        # Emit valueChanged signal for convenience (no need any extra signal)
        spin.valueChanged.connect(
            lambda val, ix=index: index.model().setData(ix, val, Qt.ItemDataRole.EditRole)
        )
        return spin

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        if value is None:
            value = 0.0
        editor.setValue(float(value))

    def setModelData(self, editor, model, index):
        editor.interpretText()  # ensure text is parsed
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)


class TsdFramesModel(QAbstractTableModel):
    """A model to handle the dict of tsdframes with checkboxes."""

    checkStateChanged = Signal(str, str, float, float, bool)

    def __init__(self, tsdframes: dict):
        super().__init__()
        self.tsdframes = tsdframes
        self.colors = GRADED_COLOR_LIST
        self.rows = [
            {
                "name": k,
                "colors": self.colors[i%len(self.colors)],
                "markersize": 10,
                "thickness": 2,
                "checked": False
            }
            for i, k in enumerate(self.tsdframes.keys())
        ]

    # ---- model dimensions ----
    def rowCount(self, parent=None):
        if parent is None:
            parent = QModelIndex()
        return len(self.rows)

    def columnCount(self, parent=None):
        return 4

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return ["TsdFrame", "Color", "Size", "Thickness"][section]

    def data(self, index, role=None):
        """What to display in the table view."""
        # Guard clause for invalid index
        # (for example, if initialize with empty tsdframe dict)
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
                return r["markersize"]
            if col == 3:
                return r["thickness"]


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
        elif index.column() == 3:
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
            return False
        row, col = index.row(), index.column()
        r = self.rows[row]

        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            # handles both  Qt.CheckState Enum and int/bool
            check_value = getattr(value, 'value', value)
            r["checked"] = (check_value == Qt.CheckState.Checked.value)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            self.checkStateChanged.emit(r["name"], r["colors"], r["markersize"], r["thickness"], r["checked"])
            return True

        if role == Qt.ItemDataRole.EditRole:
            if col == 1:
                r["colors"] = str(value)
            elif col == 2:
                r["markersize"] = float(value)
            elif col == 3:
                r["thickness"] = float(value)
            else:
                return False
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.EditRole])
            self.checkStateChanged.emit(r["name"], r["colors"], r["markersize"], r["thickness"], r["checked"])
            return True
        return False

class TsdFramesDialog(QDialog):
    """
    Dialog showing a table of tsdframe with 4 columns:
    - Column 0: name + checkbox
    - Column 1: dropdown combo
    - Column 2: number entry
    - Column 3: number entry
    """
    def __init__(self, model: TsdFramesModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("TsdFrame selection")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumSize(400, 300)

        self.view = QTableView(self)
        self.view.setModel(model)
        header = self.view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # header.setStretchLastSection(True)
        # self.view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)


        color_delegate = ComboDelegate(self.view)
        self.view.setItemDelegateForColumn(1, color_delegate)

        # Marker size
        markersize_delegate = DoubleSpinDelegate(min_=0, max_=1e12, parent=self.view)
        self.view.setItemDelegateForColumn(2, markersize_delegate)

        # Line thickness
        thickness_delegate = DoubleSpinDelegate(min_=0, max_=1e12, parent=self.view)
        self.view.setItemDelegateForColumn(3, thickness_delegate)

        layout = QVBoxLayout()

        # Add a help message
        text = ("Select the TsdFrame to superpose. \n"
                "Adjust color, marker size, and line thickness as needed. \n"
                "TsdFrame object should have even number of columns representing x,y coordinates. \n"
                "Ex : (x1, y1, x2, y2, ...) \n")
        help_label = QLabel(text)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

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



