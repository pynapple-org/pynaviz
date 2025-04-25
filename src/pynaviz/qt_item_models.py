from PyQt6.QtCore import (
    QAbstractListModel,
    QEvent,
    QItemSelectionModel,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtWidgets import QListView

import pynapple as nap

class DynamicSelectionListView(QListView):

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

            self.setSelectionMode(QListView.SelectionMode.MultiSelection)
            for row in selected_rows:
                idx = self.model().index(row, 0)
                self.selectionModel().select(
                    idx, QItemSelectionModel.SelectionFlag.Deselect
                )
            # roll back to no selection.
            self.setSelectionMode(QListView.SelectionMode.NoSelection)

            self.selectionModel().select(
                changed_index, QItemSelectionModel.SelectionFlag.Deselect
            )

        # Defer model modification safely (it crashes otherwise)
        QTimer.singleShot(0, apply_changes)

    # diversify behavior of simple click vs cmd/ctrl+click
    def mousePressEvent(self, event):
        modifiers = event.modifiers()

        if (
            modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        ):
            self.setSelectionMode(QListView.SelectionMode.MultiSelection)
        else:
            self.setSelectionMode(QListView.SelectionMode.NoSelection)

        super().mousePressEvent(event)


class ChannelListModel(QAbstractListModel):
    checkStateChanged = pyqtSignal(int)

    def __init__(self, plot):
        super().__init__()
        self._plot = plot

        if isinstance(plot.data, nap.TsGroup):
            self.checks = {i: True for i in plot.data.keys()}
        elif isinstance(plot.data, nap.TsdFrame):
            self.checks = {i: True for i in plot.data.columns}

    def rowCount(self, parent=None):
        return len(self.checks.keys())

    def data(self, index, role):
        row = index.row()
        if role == Qt.ItemDataRole.DisplayRole:
            return row
        elif role == Qt.ItemDataRole.CheckStateRole:
            return (
                Qt.CheckState.Checked if self.checks[row] else Qt.CheckState.Unchecked
            )
        return None

    def flags(self, index):
        """Flags that determines what one can do with the items."""
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsUserCheckable  # adds a check box
            | Qt.ItemFlag.ItemIsSelectable  # makes item selectable
        )

    def setData(self, index, value, role):
        if role == Qt.ItemDataRole.CheckStateRole:
            value = value.value if hasattr(value, "value") else value
            state = int(value) == Qt.CheckState.Checked.value
            if self.checks[index.row()] != state:
                self.checks[index.row()] = state
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
                self.checkStateChanged.emit(index.row())
            return True

        return False


class TsdFrameColumnListModel(QAbstractListModel):
    checkStateChanged = pyqtSignal(int)

    def __init__(self, tsdframe):
        super().__init__()
        self.data_ = tsdframe
        self.checks = {i: False for i in range(len(tsdframe.columns))}

    def rowCount(self, parent=None):
        return len(self.data_.columns)

    def data(self, index, role):
        row = index.row()
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self.data_.columns[row])
        elif role == Qt.ItemDataRole.CheckStateRole:
            return (
                Qt.CheckState.Checked if self.checks[row] else Qt.CheckState.Unchecked
            )
        return None

    def flags(self, index):
        """Flags that determines what one can do with the items."""
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsUserCheckable  # adds a check box
            | Qt.ItemFlag.ItemIsSelectable  # makes item selectable
        )

    def setData(self, index, value, role):
        if role == Qt.ItemDataRole.CheckStateRole:
            value = value.value if hasattr(value, "value") else value
            self.checks[index.row()] = int(value) == Qt.CheckState.Checked.value
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            self.checkStateChanged.emit(index.row())
            return True
        return False

    def get_selected(self):
        cols = [c for i, c in enumerate(self.data_.columns) if self.checks[i]]

        if len(cols) == 1:
            return self.data_.__class__(
                t=self.data_.index,
                d=self.data_[cols].values,
                columns=cols,
                time_support=self.data_.time_support,
                metadata=self.data_.metadata.iloc[
                    [i for i, v in self.checks.items() if v]
                ],
            )
        return self.data_[cols]


if __name__ == "__main__":
    import numpy as np
    import pynapple as nap
    from PyQt6.QtWidgets import QApplication, QListView


    my_tsdframe = nap.TsdFrame(
        t=np.arange(10),
        d=np.random.randn(10, 3),
        columns=["a", "b", "c"],
        metadata={"meta": np.array([5, 10, 15])},
    )
    app = QApplication([])
    view = DynamicSelectionListView()

    model = TsdFrameColumnListModel(my_tsdframe)
    view.setModel(model)
    model.checkStateChanged.connect(view.on_check_state_changed)
    # view.setSelectionMode(view.SelectionMode.MultiSelection)
    # view.clicked.connect(handle_click)
    view.show()

    app.exec()
    print(model.get_selected(), type(model.get_selected()))
