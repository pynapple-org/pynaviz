"""Handling of interval sets associated to base_plot classes."""
from typing import Iterable, Optional
import pynapple as nap
import pygfx


class IntervalSetInterface:
    def __init__(
            self,
            epochs: Optional[Iterable[nap.IntervalSet] | nap.IntervalSet]=None,
            labels: Optional[Iterable[str] | str]=None
    ):
        self._epochs = dict()
        if epochs is not None:
            self.add_interval_set(epochs, labels)

    def add_interval_set(self, epochs: Iterable[nap.IntervalSet] | nap.IntervalSet, labels: Optional[Iterable[str] | str]=None):
        epochs = list(epochs)
        labels = labels if labels is not None else [f"interval_{i}" for i, _ in enumerate(epochs)]
        if len(labels) != len(epochs):
            raise ValueError("The number of labels provided does not match the number of epochs.")
        self._epochs.update(dict(zip(labels, epochs)))

    def plot_intervals(self, labels: Iterable[str] | str):
        getattr(self, "material", None)


    def mesh_definition(self, color: pygfx.Color | str):
        mesh = pygfx.Mesh(
            pygfx.box_geometry(1, 2, 1),
            pygfx.MeshBasicMaterial(
                color=pygfx.Color(color), pick_write=True
            ),
        )

if __name__ == "__main__":

    import numpy as np
    import pynapple as nap
    from PyQt6.QtWidgets import QApplication
    import pynaviz as viz


    tsd1 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
    tsg = nap.TsGroup({
        i: nap.Ts(
            t=np.linspace(0, 100, 100 * (10 - i))
        ) for i in range(10)},
        metadata={
            "label": np.arange(10),
            "colors": ["hotpink", "lightpink", "cyan", "orange", "lightcoral", "lightsteelblue", "lime", "lightgreen",
                       "magenta", "pink"],
            "test": ["hello", "word"] * 5,
        }
    )
    tsdframe = nap.TsdFrame(t=np.arange(1000), d=np.random.randn(1000, 10), metadata={"label": np.random.randn(10)})
    tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))

    app = QApplication([])
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())

    # viz.TsdWidget(tsd1).show()
    # viz.TsdTensorWidget(tsdtensor).show()
    v = viz.TsGroupWidget(tsg)
    v.plot.controller.show_interval(0, 20)
    # v.plot.color_by("label", 'jet')
    # v.plot.sort_by("rate")
    v.show()
    # v = viz.TsdFrameWidget(tsdframe)
    # v.show()


    import numpy as np
    obj = IntervalSetInterface()

    mesh = pygfx.Mesh(pygfx.box_geometry(1, 2, 0),
                      pygfx.MeshBasicMaterial(
                          color=pygfx.Color("red"), pick_write=True
                      ))

    mesh.geometry.positions.data  = mesh.geometry.positions.data + np.atleast_2d(np.array([0, 10, 0]))
    print(mesh.geometry.positions.data +  np.atleast_2d(np.array([0, 10, 0])))
    disp = pygfx.Display()
    disp.show(mesh)
    #obj.mesh_definition("blue")

    app.exit(app.exec())