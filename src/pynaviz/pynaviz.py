import sys
import pynapple as nap

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QTextEdit, QPushButton, QWidget, QVBoxLayout,
    QListWidget, QHBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, QSize
from collections import defaultdict

import wgpu
import pygfx
from pygfx.renderers import WgpuRenderer


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

        # Add a QTextEdit inside the dock
        dock_content = QTextEdit()
        self.setWidget(dock_content)

        # # Add the dock to the right side of the main window
        # self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)





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
        self.setFixedWidth(self.listWidget.sizeHintForColumn(0) + 40)

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

    global QT_APP
    QT_APP = QApplication.instance()
    if QT_APP is None:
        QT_APP = QApplication(sys.argv)

    gui = GUI()

    LeftDock(pynavar, gui)

    gui.show()

    QT_APP.exit(QT_APP.exec())
    # sys.exit(QT_APP.exec())

    gui.close()
    # QT_APP.quit()

    # print("yo")

    # gc.collect()

    return