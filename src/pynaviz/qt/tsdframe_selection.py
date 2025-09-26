from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QComboBox, QPushButton, \
    QWidget, QSpinBox
from PyQt6.QtGui import QColor
from functools import partial
import sys
import pynapple as nap

from pynaviz.utils import GRADED_COLOR_LIST


class TsdFrameDialog(QDialog):
    def __init__(self, tsdframe: nap.TsdFrame, parent: QWidget, overlay_func: callable):
        super().__init__(parent)
        self.setWindowTitle("Overlay time series from TsdFrame")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(700, 300)
        self._overlay_func = overlay_func

        layout = QVBoxLayout(self)

        # Table: 5 columns (4 combos + minus)
        self.table = QTableWidget(0, 5)
        self.table.horizontalHeader().setVisible(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setHorizontalHeaderLabels(["Col x", "Col y", "Color", "Marker size", ""])
        layout.addWidget(self.table)

        # Stretch first 4 columns
        for i in range(4):
            self.table.horizontalHeader().setSectionResizeMode(i, self.table.horizontalHeader().ResizeMode.Stretch)
        self.table.setColumnWidth(4, 40)

        # Combo options
        self.columns = [
            tsdframe.columns.tolist(),
            tsdframe.columns.tolist(),
            GRADED_COLOR_LIST
        ]

        # Add row button
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Row")
        add_btn.clicked.connect(self.add_row)
        btn_layout.addWidget(add_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Add first row
        self.add_row()

    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)

        # --- Column 0–1: editable combos with placeholder ---
        for col, options in enumerate(self.columns[0:2]):
            combo = QComboBox()
            combo.setEditable(True)
            combo.lineEdit().setReadOnly(True)  # prevent typing
            combo.lineEdit().setPlaceholderText("Select…")  # grey hint
            combo.addItems([str(opt) for opt in options])
            combo.setCurrentIndex(-1)  # no default selection

            # Handler that always finds the current row/column
            def handler(text, cb=combo):
                idx = self.table.indexAt(cb.pos())
                self.on_change(idx.row(), idx.column(), text)

            combo.currentTextChanged.connect(handler)
            self.table.setCellWidget(row, col, combo)

        # --- Column 2: color combo ---
        color_combo = QComboBox()
        color_combo.addItems(self.columns[2])
        color_combo.setCurrentIndex(row % len(self.columns[2]))

        def color_handler(text, cb=color_combo):
            idx = self.table.indexAt(cb.pos())
            self.on_change(idx.row(), idx.column(), text)

        color_combo.currentTextChanged.connect(color_handler)
        self.table.setCellWidget(row, 2, color_combo)

        # --- Column 3: marker size spin box ---
        spin = QSpinBox()
        spin.setRange(1, 1000)
        spin.setValue(20)

        def spin_handler(val, sb=spin):
            idx = self.table.indexAt(sb.pos())
            self.on_change(idx.row(), idx.column(), val)

        spin.valueChanged.connect(spin_handler)
        self.table.setCellWidget(row, 3, spin)

        # --- Column 4: minus button ---
        btn = QPushButton("−")
        btn.setFixedWidth(30)
        btn.clicked.connect(partial(self.remove_row, row))
        self.table.setCellWidget(row, 4, btn)

    def remove_row(self, row):
        self.table.removeRow(row)
        # Reconnect minus buttons
        for r in range(self.table.rowCount()):
            btn = self.table.cellWidget(r, 4)
            btn.clicked.disconnect()
            btn.clicked.connect(partial(self.remove_row, r))

    def on_change(self, row, col, text):
        x_col = self.table.cellWidget(row, 0).currentText()
        y_col = self.table.cellWidget(row, 1).currentText()
        color = self.table.cellWidget(row, 2).currentText()
        size = self.table.cellWidget(row, 3).value()
        if x_col and y_col:
            self._overlay_func(x_col, y_col, color, size, True)