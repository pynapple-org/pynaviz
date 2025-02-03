import sys
import pynapple as nap
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QTextEdit, QPushButton, QWidget, QVBoxLayout,
    QListWidget, QHBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, QSize
from collections import defaultdict

from wgpu.gui.qt import WgpuCanvas
import pygfx as gfx
from pylinalg import vec_transform, vec_unproject

DOCK_TITLE_STYLESHEET = '''
    * {
        padding: 0;
        margin: 0;
        border: 0;
        background: #272822;
        color: white;
    }

    QPushButton {
        padding: 4px;
        margin: 0 1px;
    }

    QCheckBox {
        padding: 2px 4px;
        margin: 0 1px;
    }

    QLabel {
        padding: 3px;
    }

    QPushButton:hover, QCheckBox:hover {
        background: #323438;
    }

    QPushButton:pressed {
        background: #53575e;
    }

    QPushButton:checked {
        background: #6c717a;
    }
'''
DOCK_STATUS_STYLESHEET = '''
    * {
        padding: 0;
        margin: 0;
        border: 0;
        background: #272822;
        color: white;
    }

    QLabel {
        padding: 3px;
    }
'''
DOCK_LIST_STYLESHEET = '''
    * {
        border : 2px solid black;
        background : #272822;
        color : #F8F8F2;
        selection-color : yellow;
        selection-background-color : #E69F66;
    }

    QListView {
        background-color : #272822;

    }
'''

class TsdView(QDockWidget):

    def __init__(self, tsd):
        super().__init__()

        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)

        positions = np.stack((tsd.t, tsd.d, np.zeros_like(tsd))).T
        positions = positions.astype('float32')

        # Create canvas, renderer and a scene object
        self._canvas = WgpuCanvas(parent=self)
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()

        background = gfx.Background.from_color("#000")

        self._grid = gfx.Grid(
            None,
            gfx.GridMaterial(
                major_step=1,
                minor_step=0,
                thickness_space="screen",
                major_thickness=2,
                minor_thickness=0.5,
                infinite=True,
            ),
            orientation="xy",
        )
        self._grid.local.z = -1001

        self.rulerx = gfx.Ruler(tick_side="right")
        self.rulery = gfx.Ruler(tick_side="left", min_tick_distance=40)

        line = gfx.Line(
            gfx.Geometry(positions=positions),
            # gfx.LineMaterial(thickness=4.0, color="#aaf"),
        )

        self._scene.add(background, self._grid, self.rulerx, self.rulery, line)

        self._camera = gfx.OrthographicCamera(maintain_aspect=False)
        self._camera.show_rect(-100, 1100, -5, 5)

        self._controller = gfx.PanZoomController(self._camera, register_events=self._renderer)

        # # Hook up the animate callback
        self._canvas.request_draw(self.animate)
        #
        # line = gfx.Line(
        #     gfx.Geometry(positions=positions), gfx.LineMaterial(thickness=3)
        # )
        # self._scene.add(line)
        # self._canvas.update()

    def map_screen_to_world(self, pos, viewport_size):
        # first convert position to NDC
        x = pos[0] / viewport_size[0] * 2 - 1
        y = -(pos[1] / viewport_size[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        pos_ndc += vec_transform(self._camera.world.position, self._camera.camera_matrix)
        # unproject to world space
        pos_world = vec_unproject(pos_ndc[:2], self._camera.camera_matrix)

        return pos_world

    def animate(self):
        # get range of screen space
        xmin, ymin = 0, self._renderer.logical_size[1]
        xmax, ymax = self._renderer.logical_size[0], 0

        world_xmin, world_ymin, _ = self.map_screen_to_world((xmin, ymin), self._renderer.logical_size)
        world_xmax, world_ymax, _ = self.map_screen_to_world((xmax, ymax), self._renderer.logical_size)

        # set start and end positions of rulers
        self.rulerx.start_pos = world_xmin, 0, -1000
        self.rulerx.end_pos = world_xmax, 0, -1000

        self.rulerx.start_value = self.rulerx.start_pos[0]

        statsx = self.rulerx.update(self._camera, self._canvas.get_logical_size())

        self.rulery.start_pos = 0, world_ymin, -1000
        self.rulery.end_pos = 0, world_ymax, -1000

        self.rulery.start_value = self.rulery.start_pos[1]
        statsy = self.rulery.update(self._camera, self._canvas.get_logical_size())

        major_step_x, major_step_y = statsx["tick_step"], statsy["tick_step"]
        self._grid.material.major_step = major_step_x, major_step_y
        self._grid.material.minor_step = 0.2 * major_step_x, 0.2 * major_step_y

        # print(statsx)
        self._renderer.render(self._scene, self._camera)


class LeftDock(QDockWidget):

    def __init__(self, pynavar, gui, *args, **kwargs):
        super(LeftDock, self).__init__(*args, **kwargs)
        self.pynavar = pynavar
        self.gui = gui
        self.views = {}

        self.setObjectName('Variables')
        self.setWindowTitle('Variables')
        #self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)

        self.listWidget = QListWidget()
        for k in pynavar.keys():
            if k != 'data':
                self.listWidget.addItem(k)

        self.listWidget.itemDoubleClicked.connect(self.select_view)
        self.listWidget.setStyleSheet(DOCK_LIST_STYLESHEET)
        self.setWidget(self.listWidget)
        self.setFixedWidth(self.listWidget.sizeHintForColumn(0) + 50)

        self.gui.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self, Qt.Orientation.Horizontal)

        self._create_title_bar()
        self._create_status_bar()

    def select_view(self, item):
        var = self.pynavar[item.text()]

        if isinstance(var, nap.TsGroup):
            self.add_raster_view(var, item.text())
        elif isinstance(var, nap.Tsd):
            self.add_tsd_view(var, item.text())
        elif isinstance(var, nap.TsdFrame):
            self.add_tsdframe_view(var, item.text())

        return

    # def add_raster_view(self, tsgroup, name):
    #     group_times = np.hstack([tsgroup[k].index.values for k in tsgroup.keys()])
    #     group_clusters = np.hstack([
    #         np.ones(len(tsgroup[k]), dtype='int') * k for k in tsgroup.keys()
    #     ])
    #     cluster_ids = np.unique(group_clusters)
    #
    #     view = TsGroupView(group_times, group_clusters, cluster_ids=cluster_ids)
    #     view.plot()
    #     view.attach(self.gui)
    #     self.views[name] = view
    #     return

    def add_tsd_view(self, tsd, name):
        print("Adding tsd")
        view = TsdView(tsd)
        self.gui.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, view)
        view.show()
        # view.plot()
        # view.attach(self.gui)
        self.views[name] = view
        return

    # def add_tsdframe_view(self, tsdframe, name):
    #     print("TODO")
    #     return

    def _create_title_bar(self):
        """Create the title bar."""
        self._title_bar = QWidget(self)

        self._layout = QHBoxLayout(self._title_bar)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._title_bar.setStyleSheet(DOCK_TITLE_STYLESHEET)

        # Left part of the bar.
        # ---------------------

        # Widget name.
        label = QLabel(self.windowTitle())
        self._layout.addWidget(label)

        # Space.
        # ------
        self._layout.addStretch(1)

        # Layout margin.
        self._title_bar.setLayout(self._layout)
        self.setTitleBarWidget(self._title_bar)

    def _create_status_bar(self):
        # Dock has requested widget and status bar.
        widget_container = QWidget(self)
        widget_layout = QVBoxLayout(widget_container)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        widget_layout.setSpacing(0)

        widget_layout.addWidget(self.listWidget, 100)

        # Widget status text.
        self._status = QLabel('')
        self._status.setMaximumHeight(30)
        self._status.setStyleSheet(DOCK_STATUS_STYLESHEET)
        widget_layout.addWidget(self._status, 1)

        widget_container.setLayout(widget_layout)
        self.setWidget(widget_container)

class GUI(QMainWindow):
    def __init__(self):
        # HACK to ensure that closeEvent is called only twice (seems like a
        # Qt bug).
        if not QApplication.instance():  # pragma: no cover
            raise RuntimeError("A Qt application must be created.")
        super(GUI, self).__init__()

        # self.setDockOptions(QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks)
        # self.setAnimated(False)
        self.name = 'Pynaviz'
        self.setWindowTitle(self.name)
        self.setObjectName(self.name)

        self.move(200, 200)
        self.resize(QSize(1200, 800))

        self.actions = []
        self._menus = {}

        # Views,
        self._views = []
        self._view_class_indices = defaultdict(int)  # Dictionary {view_name: next_usable_index}



def get_pynapple_variables(variables=None):
    tmp = variables.copy()
    pynavar = {}
    for k, v in tmp.items():
        if hasattr(v, '__module__'):
            if "pynapple" in v.__module__ and k[0] != '_':
                pynavar[k] = v

    return pynavar


def scope(variables):

    pynavar = get_pynapple_variables(variables)

    # print(pynavar)

    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    gui = GUI()

    LeftDock(pynavar, gui)

    gui.show()

    app.exit(app.exec())
    # sys.exit(app.exec())

    gui.close()
    # app.quit()

    # print("yo")

    # gc.collect()

    return