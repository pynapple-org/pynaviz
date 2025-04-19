import pynapple as nap
from PyQt6.QtCore import QAbstractListModel, Qt

from pynaviz.utils import get_plot_attribute


class ChannelListModel(QAbstractListModel):
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
            self.checks[index.row()] = int(value) == Qt.CheckState.Checked.value
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])

            materials = get_plot_attribute(self._plot, "material")
            materials[index.row()].opacity = value
            self._plot.canvas.request_draw(self._plot.animate)
            return True

        return False


class TsdFrameColumnListModel(QAbstractListModel):
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
            self.plot.materials
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

    def handle_click(index):
        # Get current state of clicked item
        state = model.data(index, Qt.ItemDataRole.CheckStateRole)
        new_state = (
            Qt.CheckState.Unchecked
            if state == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )

        # Get selected indexes
        selected = view.selectionModel().selectedIndexes()

        # Update all selected indexes
        for idx in selected:
            model.setData(idx, new_state, Qt.ItemDataRole.CheckStateRole)

    my_tsdframe = nap.TsdFrame(
        t=np.arange(10),
        d=np.random.randn(10, 3),
        columns=["a", "b", "c"],
        metadata={"meta": np.array([5, 10, 15])},
    )
    app = QApplication([])
    view = QListView()
    model = TsdFrameColumnListModel(my_tsdframe)
    view.setModel(model)
    view.setSelectionMode(view.SelectionMode.MultiSelection)
    view.clicked.connect(handle_click)
    view.show()

    app.exec()
    print(model.get_selected(), type(model.get_selected()))
