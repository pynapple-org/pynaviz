import os
import sys
from datetime import datetime

import pynapple as nap
from PyQt6.QtCore import QSize, Qt, QTimer, QPoint, QEvent, QByteArray
from PyQt6.QtGui import QKeySequence, QShortcut, QAction, QIcon, QDoubleValidator, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget, QToolBar, QLineEdit, QFrame, QFileDialog, QSizePolicy,
)

from pynaviz.controller_group import ControllerGroup
from pynaviz.qt.widget_plot import (
    TsdFrameWidget,
    TsdTensorWidget,
    TsdWidget,
    TsGroupWidget,
    TsWidget,
)
import sys, json, base64


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

class HelpBox(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        text = (
            "Shortcuts:\n"
            "Play/Pause: Space\n"
            "Stop: S\n"
            "Next Frame: →\n"
            "Previous Frame: ←\n"
        )

        # Frameless floating box
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setFixedSize(300, 300)

        # Layout with help text
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        label.setWordWrap(True)
        layout.addWidget(label)

        # Track clicks outside
        if parent:
            parent.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            # Close the help box if clicking outside
            if not self.geometry().contains(event.globalPosition().toPoint()):
                self.close()
        return super().eventFilter(obj, event)

class MainDock(QDockWidget):

    def __init__(self, pynavar, gui, layout_path=None):
        """
        Main dock widget containing the list of variables and the play/pause button.

        Parameters
        ----------
        pynavar:
            Dictionary of pynapple variables.
        gui:
            MainWindow instance.
        layout_path:
            Path to a layout file to load on startup.
        """
        super(MainDock, self).__init__()
        self.setObjectName("MainDock")
        self.pynavar = pynavar

        # Adding controller group
        self.ctrl_group = ControllerGroup()
        self._n_dock_open = 0

        self.gui = gui
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setTitleBarWidget(QWidget())

        # --- central container widget ---
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)  # fixed position like a menu bar

        # Save action
        save_action = QAction(QIcon.fromTheme("document-save"), "Save layout", self)
        save_action.triggered.connect(self._save_layout)
        toolbar.addAction(save_action)

        # Load action
        load_action = QAction(QIcon.fromTheme("document-open"), "Load layout", self)
        load_action.triggered.connect(self._load_layout)
        toolbar.addAction(load_action)

        # Help action
        self.help_box = None
        self.help_action = QAction(QIcon.fromTheme("help-about"), "Help", self)
        self.help_action.triggered.connect(self._toggle_help_box)
        toolbar.addAction(self.help_action)
        self.help_button = toolbar.widgetForAction(self.help_action)
        layout.addWidget(toolbar)
        for action in [save_action, load_action, self.help_action]:
            button = toolbar.widgetForAction(action)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)


        # --- list widget ---
        layout.addWidget(QLabel("Variables"))
        self.listWidget = QListWidget()
        for k in pynavar.keys():
            if k != "data":
                self.listWidget.addItem(k)
        self.listWidget.itemDoubleClicked.connect(self.add_dock_widget)
        # self.listWidget.setStyleSheet(DOCK_LIST_STYLESHEET)
        layout.addWidget(self.listWidget)

        # --- Timer ---
        self.time_label = QLabel(f"{self.ctrl_group.current_time:.2f} s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.time_label.font()
        font.setPointSize(12)
        self.time_label.setFont(font)
        layout.addWidget(self.time_label)

        # --- play/pause button at bottom ---
        button_layout = QHBoxLayout()
        self.playPauseBtn = QPushButton()
        self.playPauseBtn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.playPauseBtn.setCheckable(True)
        self.playPauseBtn.toggled.connect(self._toggle_play)
        button_layout.addWidget(self.playPauseBtn)
        self.stopBtn = QPushButton()
        self.stopBtn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self.stopBtn.toggled.connect(self._stop)
        button_layout.addWidget(self.stopBtn)

        layout.addLayout(button_layout)

        self.setWidget(container)
        self.setFixedWidth(self.listWidget.sizeHintForColumn(0) + 50)

        self.gui.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self, Qt.Orientation.Horizontal
        )

        # Animation
        self.playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._play)
        self.timer.start(25) # 40 FPS

        shortcut = QShortcut(QKeySequence("Space"), self)
        shortcut.activated.connect(self._toggle_play)

        # Loading layout if provided
        if layout_path is not None and os.path.isfile(layout_path):
            self._load_layout(layout_path)

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            # Switch to pause icon
            self.playPauseBtn.setIcon(QIcon.fromTheme("media-playback-pause"))
        else:
            # Switch to play icon
            self.playPauseBtn.setIcon(QIcon.fromTheme("media-playback-start"))

    def _play(self, delta=0.025):
        if self.playing:
            self.ctrl_group.advance(delta=delta)
            self.time_label.setText(f"{self.ctrl_group.current_time:.2f} s")

    def _stop(self):
        pass

    def add_dock_widget(self, item: object) -> None:
        name = ""
        if hasattr(item, "text"):
            # item is a QListWidgetItem
            name = item.text()
        elif isinstance(item, str):
            # item is a string (variable name)
            name = item
            if item not in self.pynavar:
                print(f"Variable '{item}' not found.")
                return

        var = self.pynavar.get(name, None)
        if var is None:
            print(f"Variable '{name}' not found.")
            return
        else:
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
            label = QLabel(name)
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

            # Adding name to dock widget
            dock.setObjectName(name+"_"+str(index))

            # Adding the dock widget to the GUI window.
            self.gui.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

            # balance all right docks equally
            right_docks = [
                d for d in self.gui.findChildren(QDockWidget)
                if self.gui.dockWidgetArea(d) == Qt.DockWidgetArea.RightDockWidgetArea
            ]

            if right_docks:
                # Horizontal = width distribution (central vs right area).
                # Using larger size for the newest dock nudges the main splitter to allocate width.
                sizes_w = [1] * len(right_docks)
                sizes_w[right_docks.index(dock)] = 300  # request ~300 "units" for the new one
                # self.gui.resizeDocks(right_docks, sizes_w, Qt.Orientation.Horizontal)

                # B) Distribute HEIGHT among right-docks (stacked vertically)
                sizes_h = [1] * len(right_docks)  # equal heights
                self.gui.resizeDocks(right_docks, sizes_h, Qt.Orientation.Vertical)

            # Adding controller and render to control group
            self.ctrl_group.add(widget.plot, index)
            self._n_dock_open += 1

    def _toggle_help_box(self):
        # If the box exists and is visible, close it
        if self.help_box and self.help_box.isVisible():
            self.help_box.close()
            return

        self.help_box = HelpBox(parent=self)

        # Position it below the button
        btn_pos = self.help_button.mapToGlobal(QPoint(0, self.help_button.height()))
        self.help_box.move(btn_pos)
        self.help_box.show()

    def _save_layout(self):
        print("Saving layout...")
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M")
        default_file = os.path.join(os.getcwd(), f"layout_{dt}.json")
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Layout",
            default_file,  # suggested file name
            "Layout Files (*.json)"
        )
        if file_name:

            if not file_name.lower().endswith(".json"):
                file_name += ".json"

            geom = bytes(self.gui.saveGeometry())
            state = bytes(self.gui.saveState(version=0)) # Potentially add package versioning later

            all_docks = self.gui.findChildren(QDockWidget)
            docks = []
            order = []
            for d in all_docks:
                name = d.objectName()
                if name != "MainDock":
                    info = {"visible": d.isVisible(),
                            "floating": d.isFloating(),
                            "area": self.gui.dockWidgetArea(d).name,
                            "pos": (d.x(), d.y()),
                            "size": (d.width(), d.height()),
                            "dtype": d.widget().plot.data.__class__.__name__,
                            "varname": name.split("_")[0],
                            "index": int(name.split("_")[-1]),
                            "name": name
                            }
                    docks.append(info)
                    order.append(int(name.split("_")[-1]))
            docks = [x for _, x in sorted(zip(order, docks))]

            payload = {
                "version": 0,
                "geometry_b64": base64.b64encode(geom).decode("ascii"),
                "state_b64": base64.b64encode(state).decode("ascii"),
                "docks": docks,
            }
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Layout saved to {file_name}")

    def _load_layout(self, file_name=None):
        if file_name is None:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Layout", "", "Layout Files (*.json)")
        if file_name:
            with open(file_name, "r", encoding="utf-8") as f:
                payload = json.load(f)

            # 0) Need to empty the gui
            for d in self.gui.findChildren(QDockWidget):
                if d.objectName() != "MainDock":
                    d.close()
                    d.setParent(None)
                    d.deleteLater()

            # Clear the main window state
            self.gui.restoreState(QByteArray())
            self.gui.restoreGeometry(QByteArray())

            # Empty the controller group
            index = list(self.ctrl_group._controller_group.keys())
            for id in index:
                self.ctrl_group.remove(id)
            self._n_dock_open = 0
            assert len(self.ctrl_group._controller_group) == 0, "Controller group not empty after removing all docks."

            # 1) add every dock with the potential variable. Order them by number in the name
            for widget in payload.get("docks", []):
                if widget['varname'] in self.pynavar:
                    self.add_dock_widget(widget['varname'])
                assert self.gui.findChild(QDockWidget, widget['name']) is not None, f"Dock {widget['name']} was not created."

            # 2) Restore geometry first, then layout
            geom = QByteArray.fromBase64(payload["geometry_b64"].encode("ascii"))
            state = QByteArray.fromBase64(payload["state_b64"].encode("ascii"))

            self.gui.restoreGeometry(geom)
            ok = self.gui.restoreState(state, payload.get("version", 0))
            print("restoreState ok:", ok)


class MainWindow(QMainWindow):
    def __init__(self):
        # HACK to ensure that closeEvent is called only twice (seems like a
        # Qt bug).
        if not QApplication.instance():  # pragma: no cover
            raise RuntimeError("A Qt application must be created.")
        super().__init__()

        self.name = "Pynaviz"
        self.setWindowTitle(self.name)
        self.setObjectName(self.name)
        self.resize(QSize(1200, 800))
        # font = QFont("Arial", 10)  # Font family Arial, size 14 pt
        # self.setFont(font)
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


def scope(variables, layout_path=None):

    # Filtering for pynapple variables
    pynavar = get_pynapple_variables(variables)

    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    gui = MainWindow()

    MainDock(pynavar, gui, layout_path=layout_path)

    gui.show()

    app.exit(app.exec())

    gui.close()

    return
