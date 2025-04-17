from PyQt6.QtCore import Qt, QAbstractListModel


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
            return Qt.CheckState.Checked if self.checks[row] else Qt.CheckState.Unchecked
        return None

    def flags(self, index):
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsSelectable
        )

    def setData(self, index, value, role):
        if role == Qt.ItemDataRole.CheckStateRole:
            self.checks[index.row()] = (int(value) == Qt.CheckState.Checked.value)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
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
                metadata=self.data_.metadata.iloc[[i for i, v in self.checks.items() if v]],
            )
        return self.data_[cols]


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtWidgets import QListView
    import pynapple as nap
    import numpy as np

    my_tsdframe = nap.TsdFrame(t=np.arange(10), d=np.random.randn(10,3), columns=["a", "b", "c"], metadata={"meta":np.array([5, 10, 15])})
    app = QApplication([])
    view = QListView()
    model = TsdFrameColumnListModel(my_tsdframe)
    view.setModel(model)
    view.show()

    app.exec()
    print(model.get_selected(), type(model.get_selected()))