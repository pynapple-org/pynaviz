import sys

import pynapple as nap
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStyle,
    QWidget,
)

from .controller import ControllerGroup
from .widget_plot import (
    TsdFrameWidget,
    TsdTensorWidget,
    TsdWidget,
    TsGroupWidget,
    TsWidget,
)

DOCK_TITLE_STYLESHEET = """
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
"""
DOCK_LIST_STYLESHEET = """
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
"""


class ListDock(QDockWidget):

    def __init__(self, pynavar, gui):
        super(ListDock, self).__init__()
        self.pynavar = pynavar
        self.gui = gui

        self.setObjectName("Variables")
        self.setWindowTitle("Variables")
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)

        self.listWidget = QListWidget()
        for k in pynavar.keys():
            if k != "data":
                self.listWidget.addItem(k)

        self.listWidget.itemDoubleClicked.connect(self.add_dock_widget)
        self.listWidget.setStyleSheet(DOCK_LIST_STYLESHEET)
        self.setWidget(self.listWidget)
        self.setFixedWidth(self.listWidget.sizeHintForColumn(0) + 50)

        self.gui.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self, Qt.Orientation.Horizontal
        )

        self._create_title_bar()

        # Adding controller group
        self.ctrl_group = ControllerGroup()
        self._n_dock_open = 0

    def add_dock_widget(self, item):
        var = self.pynavar[item.text()]

        index = self._n_dock_open

        # Getting the pynaviz widget class
        if isinstance(var, nap.TsGroup):
            widget = TsGroupWidget(var, index=index, set_parent=True)
        elif isinstance(var, nap.Tsd):
            widget = TsdWidget(var, index=index, set_parent=True)
        elif isinstance(var, nap.TsdFrame):
            widget = TsdFrameWidget(var, index=index, set_parent=True)
        elif isinstance(var, nap.TsdTensor):
            widget = TsdTensorWidget(var, index=index, set_parent=True)
        elif isinstance(var, nap.Ts):
            widget = TsWidget(var, index=index, set_parent=True)

        # Instantiating the dock widget
        dock = QDockWidget()
        dock.setWidget(widget)

        # Adding the name of the variable to the button container
        layout = widget.button_container.layout()
        label = QLabel(item.text())
        # label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(label)
        layout.addStretch()

        # Adding the close button to the button container
        close_btn = QPushButton()
        close_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        )
        close_btn.setFixedSize(12, 12)
        close_btn.clicked.connect(dock.close)
        layout.addWidget(close_btn)

        # Setting the button container as the title bar
        dock.setTitleBarWidget(widget.button_container)

        # Adding the dock widget to the GUI window.
        self.gui.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # Adding controller and render to control group
        self.ctrl_group.add(widget.plot.controller, widget.plot.renderer, index)
        self._n_dock_open += 1

    def _create_title_bar(self):
        """Create the title bar."""
        self._title_bar = QWidget(self)
        self._layout = QHBoxLayout(self._title_bar)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._title_bar.setStyleSheet(DOCK_TITLE_STYLESHEET)
        label = QLabel(self.windowTitle())
        self._layout.addWidget(label)
        self._layout.addStretch(1)
        self._title_bar.setLayout(self._layout)
        self.setTitleBarWidget(self._title_bar)


class GUI(QMainWindow):
    def __init__(self):
        # HACK to ensure that closeEvent is called only twice (seems like a
        # Qt bug).
        if not QApplication.instance():  # pragma: no cover
            raise RuntimeError("A Qt application must be created.")
        super(GUI, self).__init__()

        self.name = "Pynaviz"
        self.setWindowTitle(self.name)
        self.setObjectName(self.name)
        self.resize(QSize(1200, 800))

        # Enable nested docking (so docks can be stacked)
        self.setDockNestingEnabled(True)


def get_pynapple_variables(variables=None):
    tmp = variables.copy()
    pynavar = {}
    for k, v in tmp.items():
        if hasattr(v, "__module__"):
            if "pynapple" in v.__module__ and k[0] != "_":
                pynavar[k] = v

    return pynavar


def scope(variables):

    pynavar = get_pynapple_variables(variables)

    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    gui = GUI()

    ListDock(pynavar, gui)

    gui.show()

    app.exit(app.exec())

    gui.close()

    return
