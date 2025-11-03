from typing import Any

import pynapple as nap
from PyQt6.QtCore import (
    QAbstractListModel,
    QEvent,
    QItemSelectionModel,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtWidgets import QDialog, QListView, QVBoxLayout, QWidget


class DynamicSelectionListView(QListView):
    """A QListView that allows dynamic selection of multiple items with checkboxes."""

    def on_check_state_changed(self, changed_row):
        selected_rows = [
            index.row() for index in self.selectionModel().selectedIndexes()
        ]

        if not selected_rows:
            return  # nothing selected, nothing to do

        changed_index = self.model().index(changed_row, 0)
        new_state = self.model().data(changed_index, Qt.ItemDataRole.CheckStateRole)

        def apply_changes():
            for row in selected_rows:
                idx = self.model().index(row, 0)
                self.model().setData(idx, new_state, Qt.ItemDataRole.CheckStateRole)

            for row in selected_rows:
                idx = self.model().index(row, 0)
                self.selectionModel().select(
                    idx, QItemSelectionModel.SelectionFlag.Deselect
                )

            self.selectionModel().select(
                changed_index, QItemSelectionModel.SelectionFlag.Deselect
            )

        # Defer model modification safely (it crashes otherwise)
        QTimer.singleShot(0, apply_changes)

    def selectionCommand(self, index, event=None):
        if event is not None and event.type() == QEvent.Type.MouseButtonPress:
            modifiers = event.modifiers()

            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                return super().selectionCommand(index, event)  # allow range selection

            if modifiers & (
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
            ):
                return (
                    QItemSelectionModel.SelectionFlag.Toggle
                )  # Cmd/Ctrl + click → toggle

            # Regular click
            if self.selectionModel().isSelected(index):
                return QItemSelectionModel.SelectionFlag.Deselect
            else:
                return QItemSelectionModel.SelectionFlag.Select

        return super().selectionCommand(index, event)


class ChannelListModel(QAbstractListModel):
    """A model to handle the list of channels with checkboxes."""

    checkStateChanged = pyqtSignal(int)

    def __init__(self, data: Any):
        super().__init__()

        if isinstance(data, nap.TsGroup):
            self.checks = {i: True for i in data.keys()}
            self.names = list(data.keys())
        elif isinstance(data, nap.TsdFrame):
            self.checks = {i: True for i in data.columns}
            self.names = list(data.columns)
        elif isinstance(data, nap.IntervalSet):
            self.checks = {i: True for i in data.index}
            self.names = list(data.index)
        elif isinstance(data, dict) and all(
            isinstance(v, nap.IntervalSet) for v in data.values()
        ):
            self.checks = {i: False for i in range(len(data))}
            self.names = list(range(len(data)))

    def rowCount(self, parent=None):
        return len(self.checks.keys())

    def flags(self, index):
        """Flags that determines what one can do with the items."""
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsUserCheckable  # adds a check box
            | Qt.ItemFlag.ItemIsSelectable  # makes item selectable
        )

    def data(self, index, role=None):
        """What to display in the list view."""
        row = index.row()
        if role == Qt.ItemDataRole.DisplayRole:
            return self.names[row]
        elif role == Qt.ItemDataRole.CheckStateRole:
            return (
                Qt.CheckState.Checked if self.checks[self.names[row]] else Qt.CheckState.Unchecked
            )
        return None

    def setData(self, index, value, role):
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
        if role == Qt.ItemDataRole.CheckStateRole:
            value = value.value if hasattr(value, "value") else value
            state = int(value) == Qt.CheckState.Checked.value
            if self.checks[self.names[index.row()]] != state:
                self.checks[self.names[index.row()]] = state
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
                self.checkStateChanged.emit(index.row())
            return True

        return False


class ChannelList(QDialog):

    checkStateChanged = pyqtSignal(int)

    """
    A dialog listing selectable channels (e.g., for visibility toggling).

    Parameters
    ----------
    model : ChannelListModel
        Data model that holds the list of channel states.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, model: ChannelListModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Channel List")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setFixedSize(300, 150)

        self.view = DynamicSelectionListView(self)
        self.view.setSelectionMode(self.view.SelectionMode.ExtendedSelection)
        self.view.setModel(model)
        model.checkStateChanged.connect(self.view.on_check_state_changed)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)





#######################
# Unused for now, but might be useful later
#######################

#
#
# class TsdFrameColumnListModel(QAbstractListModel):
#     checkStateChanged = pyqtSignal(int)
#
#     def __init__(self, tsdframe):
#         super().__init__()
#         self.data_ = tsdframe
#         self.checks = {i: False for i in range(len(tsdframe.columns))}
#
#     def rowCount(self, parent=None):
#         return len(self.data_.columns)
#
#     def data(self, index, role):
#         row = index.row()
#         if role == Qt.ItemDataRole.DisplayRole:
#             return str(self.data_.columns[row])
#         elif role == Qt.ItemDataRole.CheckStateRole:
#             return (
#                 Qt.CheckState.Checked if self.checks[row] else Qt.CheckState.Unchecked
#             )
#         return None
#
#     def flags(self, index):
#         """Flags that determines what one can do with the items."""
#         return (
#             Qt.ItemFlag.ItemIsEnabled
#             | Qt.ItemFlag.ItemIsUserCheckable  # adds a check box
#             | Qt.ItemFlag.ItemIsSelectable  # makes item selectable
#         )
#
#     def setData(self, index, value, role):
#         if role == Qt.ItemDataRole.CheckStateRole:
#             value = value.value if hasattr(value, "value") else value
#             self.checks[index.row()] = int(value) == Qt.CheckState.Checked.value
#             self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
#             self.checkStateChanged.emit(index.row())
#             return True
#         return False
#
#     def get_selected(self):
#         cols = [c for i, c in enumerate(self.data_.columns) if self.checks[i]]
#
#         if len(cols) == 1:
#             return self.data_.__class__(
#                 t=self.data_.index,
#                 d=self.data_[cols].values,
#                 columns=cols,
#                 time_support=self.data_.time_support,
#                 metadata=self.data_.metadata.iloc[
#                     [i for i, v in self.checks.items() if v]
#                 ],
#             )
#         return self.data_[cols]
#
#
# if __name__ == "__main__":
#     import numpy as np
#     import pynapple as nap
#     from PyQt6.QtWidgets import QApplication, QListView
#
#     my_tsdframe = nap.TsdFrame(
#         t=np.arange(10),
#         d=np.random.randn(10, 3),
#         columns=["a", "b", "c"],
#         metadata={"meta": np.array([5, 10, 15])},
#     )
#     app = QApplication([])
#     view = DynamicSelectionListView()
#
#     model = TsdFrameColumnListModel(my_tsdframe)
#     view.setModel(model)
#     view.setSelectionMode(view.SelectionMode.ExtendedSelection)
#     model.checkStateChanged.connect(view.on_check_state_changed)
#     # view.setSelectionMode(view.SelectionMode.MultiSelection)
#     # view.clicked.connect(handle_click)
#     view.show()
#
#     app.exec()
#     print(model.get_selected(), type(model.get_selected()))
