import base64
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Union

import pynapple as nap
from PyQt6.QtCore import QByteArray, QEvent, QPoint, QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStyle,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..audiovideo import PlotVideo
from ..audiovideo.video_plot import PlotBaseVideoTensor
from ..controller_group import ControllerGroup
from .widget_plot import (
    IntervalSetWidget,
    TsdFrameWidget,
    TsdTensorWidget,
    TsdWidget,
    TsGroupWidget,
    TsWidget,
    VideoWidget,
)

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


@dataclass
class NWBReference:
    """Store reference to NWB file object and variable key.

    Keeps the NWB file (and its IO handle) alive to prevent HDF5 datasets
    from being closed during garbage collection.
    """
    nwb_file: nap.io.NWBFile
    key: str


def get_children_dict(parent: QTreeWidget | QTreeWidgetItem):
    """Helper function to get children as a dictionary"""
    children = {}

    if isinstance(parent, QTreeWidget):
        count = parent.topLevelItemCount()
        for i in range(count):
            child = parent.topLevelItem(i)
            children[child.text(0)] = child
    else:
        count = parent.childCount()
        for i in range(count):
            child = parent.child(i)
            children[child.text(0)] = child

    return children

def _get_item_key_path(item: QTreeWidgetItem, key_path:None|list=None) -> list:
    if key_path is None:
        key_path = [item.text(0)]
    parent = item.parent()
    if parent is not None:
        key_path.append(parent.text(0))
        key_path = _get_item_key_path(parent, key_path)
    else:
        key_path = key_path[::-1]
    return key_path

def _get_variable_from_key_path(variables: dict, key_path: list[str]):
    var = variables
    for key in key_path:
        var = var.get(key, None)
        if var is None:
            break
    return var

class HelpBox(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        text = (
            "Add variables by double-clicking them in the list.\n"
            "Specific plot controls are available in each dock's menu (top-left corner).\n\n"
            "Shortcuts:\n"
            "Play/Pause: Space\n"
            "Save layout: Ctrl+s\n"
            "Load layout: Ctrl+o\n"
            "Reset view: r\n\n"
            "Specific to TsdFrame:\n"
            "Increase contrast: i\n"
            "Decrease contrast: d\n\n"
            "Specific to IntervalSet & Timestamps:\n"
            "Jump to next interval/timestamp: n or ->\n"
            "Jump to previous interval/timestamp: p or <-\n"
        )

        # Frameless floating box
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setFixedSize(800, 600)

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

    def __init__(self, variables, gui, layout_path=None):
        """
        Main dock widget containing the list of variables and the play/pause button.

        Parameters
        ----------
        variables:
            Dictionary of pynapple variables.
        gui:
            MainWindow instance.
        layout_path:
            Path to a layout file to load on startup.
        """
        super(MainDock, self).__init__()
        self.setObjectName("MainDock")
        self.variables = variables
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
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        toolbar.addAction(save_action)

        # Load action
        load_action = QAction(QIcon.fromTheme("document-open"), "Load layout", self)
        load_action.triggered.connect(self._load_layout)
        load_action.setShortcut(QKeySequence("Ctrl+O"))
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
            button.setFixedWidth(25)


        # --- list widget ---
        self._interval_set_key_paths = []
        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderLabel("Variables")
        self.treeWidget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self._add_items_to_tree_widget(variables)

        layout.addWidget(self.treeWidget)

        # --- Timer ---
        time_layout = QHBoxLayout()
        self.time_spin_box = QDoubleSpinBox()
        self.time_spin_box.setStyleSheet("font-size: 10pt;")
        self.time_spin_box.setMinimum(0)
        self.time_spin_box.setMaximum(1)
        self.time_spin_box.setValue(0.5)
        self.time_spin_box.valueChanged.connect(
            self._on_spinbox_changed
        )
        time_layout.addWidget(self.time_spin_box, 1)  # stretch factor = 1

        self.time_unit_combo = QComboBox()
        self.time_unit_combo.setStyleSheet("font-size: 10pt;")
        self.time_unit_combo.addItem('us', 1e6)  # text, data
        self.time_unit_combo.addItem('ms', 1e3)
        self.time_unit_combo.addItem('s', 1.0)
        self.time_unit_combo.setCurrentIndex(2)  # default to seconds
        self.time_unit_combo.setFixedWidth(55)
        self.time_unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        time_layout.addWidget(self.time_unit_combo, 0)  # stretch factor = 0

        time_layout.addStretch()
        layout.addLayout(time_layout)


        # self.time_label = QLabel("0.0 s")
        # self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # font = self.time_label.font()
        # font.setPointSize(12)
        # self.time_label.setFont(font)
        # layout.addWidget(self.time_label)

        # --- play/pause button at bottom ---
        button_layout = QHBoxLayout()
        self.skipBackwardBtn = QPushButton()
        self.skipBackwardBtn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward)
        )
        self.skipBackwardBtn.clicked.connect(self._skip_backward)
        button_layout.addWidget(self.skipBackwardBtn)
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
        self.stopBtn.clicked.connect(self._stop)
        button_layout.addWidget(self.stopBtn)
        self.skipForwardBtn = QPushButton()
        self.skipForwardBtn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward)
        )
        self.skipForwardBtn.clicked.connect(self._skip_forward)
        button_layout.addWidget(self.skipForwardBtn)

        layout.addLayout(button_layout)

        self.setWidget(container)
        self.setFixedWidth(
            max(
                self.treeWidget.sizeHintForColumn(0) + 50,
                toolbar.sizeHint().width() + 20
            )
        )

        self.gui.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self, Qt.Orientation.Horizontal
        )

        # Animation
        self.playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._play)

        shortcut = QShortcut(QKeySequence("Space"), self)
        shortcut.activated.connect(self._toggle_play)

        # Adding controller group
        self.ctrl_group = ControllerGroup(callback=self._update_time_label)

        # Loading layout if provided
        if layout_path is not None and os.path.isfile(layout_path):
            self._restore_layout(layout_path)

    def on_item_double_clicked(self, item, column):
        """Handle double-click only for leaf items"""
        if item.childCount() == 0:  # This is a leaf item
            self.add_dock_widget(item)

    def _add_items_to_tree_widget(self, item_dict: dict, parent: None | QTreeWidgetItem = None, clear: bool = False):
        if clear:
            self.treeWidget.clear()
            self._interval_set_key_paths = []

        if parent is None:
            parent = self.treeWidget

        # Get existing children
        import pdb
        pdb.set_trace()
        children = get_children_dict(parent)

        for key, value in item_dict.items():
            if key in children:
                child = children[key]
                # Make sure it is another sub-tree with elements
                assert isinstance(value, dict), ("Variable name for vars at the same level of a tree must be unique."
                                                 f"\nName '{key}' is not unique")
                self._add_items_to_tree_widget(value, parent=child, clear=False)
            else:
                # Create new item
                item = QTreeWidgetItem(parent, [key])
                if isinstance(value, dict):
                    self._add_items_to_tree_widget(value, parent=item, clear=False)
                elif isinstance(value, nap.IntervalSet):
                    self._interval_set_key_paths.append(_get_item_key_path(item))

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            # Switch to pause icon
            self.timer.start(25)  # 40 FPS
            self.playPauseBtn.setIcon(QIcon.fromTheme("media-playback-pause"))
        else:
            # Switch to play icon
            self.timer.stop()
            self.playPauseBtn.setIcon(QIcon.fromTheme("media-playback-start"))

    def _play(self, delta=0.025):
        self.ctrl_group.advance(delta=delta)
        self._update_time_label(self.ctrl_group.current_time)

    def _on_unit_changed(self):
        """When user changes units, update the spinbox display"""
        multiplier = self.time_unit_combo.currentData()
        display_value = self.ctrl_group.current_time * multiplier
        self.time_spin_box.blockSignals(True)  # Prevent recursion
        self.time_spin_box.setValue(display_value)
        self.time_spin_box.blockSignals(False)

    def _on_spinbox_changed(self, value: float):
        """Handle spinbox changes based on source enum"""
        multiplier = self.time_unit_combo.currentData()
        self.ctrl_group.set_interval(value / multiplier, None)


    def _update_time_label(self, current_time):
        time_multiplier = self.time_unit_combo.currentData()
        self.time_spin_box.blockSignals(True)
        min_time, max_time = self._get_max_min_time()
        if max_time != -float("inf") and min_time != float("inf"):
            self.time_spin_box.setMinimum(min_time * time_multiplier)
            self.time_spin_box.setMaximum(max_time * time_multiplier)
        self.time_spin_box.setValue(time_multiplier * current_time)
        self.time_spin_box.blockSignals(False)


    def _stop(self):
        if self.playing:
            self._toggle_play()
        self.ctrl_group.set_interval(0, 1)
        self._update_time_label(self.ctrl_group.current_time)

    def _get_max_min_time(self) -> tuple[float, float]:
        max_time = -float("inf")
        min_time = float("inf")
        for dock_widget in self.gui.findChildren(QDockWidget):
            # if not isinstance(dock_widget, QWidget):
            base_plot = getattr(dock_widget.widget(), "plot", None)
            if base_plot is not None:
                data = base_plot.data
                if hasattr(data, "time_support"):
                    mn = data.time_support.start[0]
                    mx = data.time_support.end[-1]
                elif isinstance(data, nap.IntervalSet):
                    mn = data.start[0]
                    mx = data.end[-1]
                else:
                    mn = getattr(base_plot.data.index, "values", base_plot.data.index)[0]
                    mx = getattr(base_plot.data.index, "values", base_plot.data.index)[-1]
                min_time = min(min_time, mn)
                max_time = max(max_time, mx)
        return min_time, max_time

    def _skip_backward(self):
        min_time, _ = self._get_max_min_time()
        if min_time != -float("inf"):
            width = None
            for ctrl in self.ctrl_group._controller_group.values():
                if hasattr(ctrl, "get_xlim"):
                    xlim = ctrl.get_xlim()
                    width = xlim[1] - xlim[0]
                    break
            self.ctrl_group.set_interval(min_time, min_time + width)
            self._update_time_label(self.ctrl_group.current_time)


    def _skip_forward(self):
        _, max_time = self._get_max_min_time()
        if max_time != float("inf"):
            self.ctrl_group.set_interval(max_time, None)
            self._update_time_label(self.ctrl_group.current_time)


    def add_dock_widget(self, item: QTreeWidgetItem | list[str]) -> None:
        key_path = self._extract_variable_path(item)
        if not key_path:
            return

        var = self._get_variable(key_path)
        if var is None:
            return

        widget = self._create_widget_for_variable(var)
        if widget is None:
            return
        widget_name = "/".join(key_path)
        dock = self._create_dock(widget_name, widget, key_path)
        self._add_dock_to_gui(dock)
        self._register_controller(widget)
        self._n_dock_open += 1
        min_time, max_time = self._get_max_min_time()
        if max_time != -float("inf") and min_time != float("inf"):
            time_multiplier = self.time_unit_combo.currentData()
            self.time_spin_box.setMinimum(min_time * time_multiplier)
            self.time_spin_box.setMaximum(max_time * time_multiplier)

    @staticmethod
    def _extract_variable_path(item: QTreeWidgetItem | list[str]) -> list[str] | None:
        """Return the variable name from a QListWidgetItem or a string."""

        if isinstance(item, QTreeWidgetItem):
            return _get_item_key_path(item)
        elif isinstance(item, list):
            return item
        else:
            print("Invalid item type for dock widget.")
            return None

    def _get_variable(self, key_path: list[str]):
        """Fetch the variable from variables and validate."""
        var = _get_variable_from_key_path(self.variables, key_path)
        if var is None:
            print(f"Variable {'/'.join(key_path)} not found.")
        return var

    def _create_widget_for_variable(self, var) -> object | None:
        """Return the correct widget based on the variable type."""
        # TODO: grab iset only from the same tree level.
        index = self._n_dock_open
        if isinstance(var, nap.TsGroup):
            interval_sets = {'/'.join(k): _get_variable_from_key_path(self.variables, k) for k in self._interval_set_key_paths}
            return TsGroupWidget(var, index=index, set_parent=True, interval_sets=interval_sets)
        elif isinstance(var, nap.Tsd):
            interval_sets = {'/'.join(k): _get_variable_from_key_path(self.variables, k) for k in self._interval_set_key_paths}
            return TsdWidget(var, index=index, set_parent=True, interval_sets=interval_sets)
        elif isinstance(var, nap.TsdFrame):
            interval_sets = {'/'.join(k): _get_variable_from_key_path(self.variables, k) for k in self._interval_set_key_paths}
            return TsdFrameWidget(var, index=index, set_parent=True, interval_sets=interval_sets)
        elif isinstance(var, nap.TsdTensor):
            return TsdTensorWidget(var, index=index, set_parent=True)
        elif isinstance(var, nap.Ts):
            return TsWidget(var, index=index, set_parent=True)
        elif isinstance(var, nap.IntervalSet):
            return IntervalSetWidget(var, index=index, set_parent=True)
        elif isinstance(var, NWBReference):
            var = var.nwb_file[var.key]
            return self._create_widget_for_variable(var)
        elif isinstance(var, VideoWidget):
            return var  # already a widget
        else:
            print(f"Variable of type '{type(var)}' is not supported for plotting.")
            return None

    def _create_dock(self, name: str, widget: object, key_path: list[str]) -> QDockWidget:
        """Create and configure the QDockWidget with title bar and controls."""
        dock = QDockWidget()
        dock.setWidget(widget)
        dock.setObjectName(f"{name}_{self._n_dock_open}")
        dock.setProperty("key_path", key_path)

        # Add name and close button to the widget's button container
        layout = widget.button_container.layout()
        label = QLabel(name)
        layout.addWidget(label)
        layout.addStretch()

        # Connect close button with cleanup
        close_btn = QPushButton()
        close_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
        close_btn.setFixedSize(15, 15)
        close_btn.clicked.connect(lambda: self._cleanup_and_close_dock(dock))

        layout.addWidget(close_btn)
        widget.button_container.setMinimumHeight(15)
        dock.setTitleBarWidget(widget.button_container)
        return dock

    def _cleanup_and_close_dock(self, dock):
        """Properly clean up and close a dock widget"""
        # Remove from controller group if registered
        widget = dock.widget()
        if hasattr(widget, 'plot'):
            # remove from controls
            ctrl_id = widget.plot.controller._controller_id
            self.ctrl_group.remove(ctrl_id)
        dock.deleteLater()

    def _add_dock_to_gui(self, dock: QDockWidget) -> None:
        """Add dock to the GUI and balance right docks vertically."""
        self.gui.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # Balance heights of all right docks
        right_docks = [
            d for d in self.gui.findChildren(QDockWidget)
            if self.gui.dockWidgetArea(d) == Qt.DockWidgetArea.RightDockWidgetArea
        ]
        if right_docks:
            sizes_h = [1] * len(right_docks)
            self.gui.resizeDocks(right_docks, sizes_h, Qt.Orientation.Vertical)

    def _register_controller(self, widget: object) -> None:
        """Register the widget's plot in the controller group."""
        self.ctrl_group.add(widget.plot, self._n_dock_open)

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
                            "key_path": d.property("key_path"),
                            "index": int(name.split("_")[-1]),
                            "name": name
                            }
                    print(info)
                    docks.append(info)
                    order.append(int(name.split("_")[-1]))
            docks = [x for _, x in sorted(zip(order, docks))]

            payload = {
                "version": 0,
                "geometry_b64": base64.b64encode(geom).decode("ascii"),
                "state_b64": base64.b64encode(state).decode("ascii"),
                "docks": docks,
                "file_paths": list(self.gui._open_file_paths),
            }
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Layout saved to {file_name}")

    def _restore_layout(self, file_name):
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

        # 1) Reload the files
        file_paths = payload.get("file_paths", [])
        self.gui._load_multiple_files(file_paths)

        # 2) add every dock with the potential variable. Order them by number in the name
        for widget in payload.get("docks", []):
            var = _get_variable_from_key_path(self.variables, widget['key_path'])
            if var is not None:
                print(f"Adding var from path {widget['key_path']}.")
                self.add_dock_widget(widget['key_path'])

            # Note: simply matching the widget name stored in the payload results in a bug
            # when the following happens:
            # 1. Open > 1 docks
            # 2. Close any dock but the last -> now the last dock name is varname_N but N-1 docks are opened
            # 3. Save layout and close, this will store widget["name"] == varname_N
            # 4. Try to reopen loading the layout.
            #    The last dock name will be varname_{N-1} but widget["name"] is varname_N
            # To prevent this, adjust the index.
            dock_name = re.sub(r"_\d+$", f"_{self._n_dock_open-1}", widget['name'])
            assert self.gui.findChild(QDockWidget,
                                      dock_name) is not None, f"Dock {dock_name} was not created."

        # 3) Restore geometry first, then layout
        geom = QByteArray.fromBase64(payload["geometry_b64"].encode("ascii"))
        state = QByteArray.fromBase64(payload["state_b64"].encode("ascii"))

        self.gui.restoreGeometry(geom)
        ok = self.gui.restoreState(state, payload.get("version", 0))
        print("restoreState ok:", ok)

    def _load_layout(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Layout", "", "Layout Files (*.json)")
        if file_name:
            self._restore_layout(file_name)


class MainWindow(QMainWindow):

    _file_extensions = {
            "Pynapple": [".npz"],
            "NWB": [".nwb"],
            "Video": [".avi", ".mp4", ".mkv"],
            # uncomment above when PlotAudio is available.
            # "Audio": [".mp3", ".wav", ".flac"]
        }

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
        self.create_menu_bar()
        self._open_file_paths = set()

    def create_menu_bar(self):
        menubar = self.menuBar()

        # Create File menu
        file_menu = menubar.addMenu("&File")

        # Open action
        open_action = QAction("&Open File...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)  # Ctrl+O
        open_action.setStatusTip("Open an existing file")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

    def open_file(self):
        extensions = self._file_extensions
        # create the formatted string for QFileDialog ext
        ext_string = "All ("
        for ext in (v for exts in extensions.values() for v in exts):
            ext_string += f"*{ext} "
        ext_string = ext_string[:-1] + ");;"

        for ext_name, ext_list in extensions.items():
            ext_string += ext_name + " ("
            for ext in ext_list:
                ext_string += f"*{ext} "
            ext_string = ext_string[:-1] + ");;"
        ext_string = ext_string[:-2]

        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Files",
            "",
            ext_string
        )
        self._load_multiple_files(filenames)

    def _load_multiple_files(self, filenames: list[str]):
        # find main dock
        dock = self.findChildren(MainDock)
        if len(dock) < 1:
            raise RuntimeError("MainWindow should have at least one dock.")
        dock_widget = dock[0]

        # create the variable dict
        def get_type(name: pathlib.Path) -> None | Literal["Pynapple", "NWB", "Video"]:
            file_type = None
            for tp, exts in self._file_extensions.items():
                if name.suffix in exts:
                    file_type = tp
                    break
            return file_type
        variables = dock_widget.variables
        new_vars = {}
        for name in filenames:
            name = pathlib.Path(name)
            file_type = get_type(name)
            if file_type is None:
                print(f"File type {pathlib.Path(name).suffix} not supported. Skipping.")
            elif file_type in ["Pynapple"]:
                new_vars.update({name.stem + name.suffix: nap.load_file(name)})
                self._open_file_paths.add(name.as_posix())
            elif file_type in ["NWB"]:
                data: nap.NWBFile = nap.load_file(name)
                nap_obj_dict = {}
                for key in data.keys():
                    nap_obj_dict[key] = NWBReference(nwb_file=data, key=key)
                new_vars.update({name.stem + name.suffix: nap_obj_dict})
                self._open_file_paths.add(name.as_posix())
            elif file_type == "Video":
                new_vars.update({name: PlotVideo(name)})
                self._open_file_paths.add(name.as_posix())
            else:
                raise TypeError(f"Developer forgot to add file type `{file_type}` to the loader.")
        variables.update(new_vars)
        dock_widget._add_items_to_tree_widget(new_vars)


def get_pynapple_variables(
    variables: dict | list | tuple | None = None
) -> dict:
    if isinstance(variables, (list, tuple)):
        # Try to get variable names from the calling frame
        import inspect
        frame = inspect.currentframe().f_back
        names = [name for name, val in frame.f_locals.items() if val is variables]
        var_name = names[0] if names else "var"
        variables = {f"{var_name}_{i}": v for i, v in enumerate(variables)}
    elif isinstance(variables, dict):
        variables = variables.copy()
    else:
        return {}

    result: dict = {}
    for k, v in variables.items():
        if hasattr(v, "__module__"):
            if "pynapple" in v.__module__ and k[0] != "_":
                result[k] = v

            if "pynaviz" in v.__module__ and k[0] != "_" and isinstance(v, PlotBaseVideoTensor):
                result[k] = v

    return result

def scope(variables: Union[dict, list, tuple], layout_path: str = None):

    # Filtering for pynapple variables
    variables = get_pynapple_variables(variables)

    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    gui = MainWindow()

    MainDock(variables, gui, layout_path=layout_path)

    gui.show()

    app.exit(app.exec())

    gui.close()

    return
