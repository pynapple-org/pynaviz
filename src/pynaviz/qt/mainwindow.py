import os
import pathlib
import sys
from typing import Any, Literal, Union

import pynapple as nap
from PySide6.QtCore import QByteArray, QEvent, QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QStyle,
    QWidget,
)

from ..controller_group import ControllerGroup
from .icons import icon_base64
from .layout_manager import LayoutManagerMixin
from .references import EphysReference, NWBReference
from .variable_dock import VariableDock, _get_variable_from_key_path
from .variable_loader import get_pynapple_variables
from .widget_plot import (
    IntervalSetWidget,
    TsdFrameWidget,
    TsdTensorWidget,
    TsdWidget,
    TsGroupWidget,
    TsWidget,
    VideoHandler,
    VideoWidget,
)


class MainWindow(LayoutManagerMixin, QMainWindow):
    """Main application window for pynaviz.

    ``MainWindow`` is the top-level Qt widget that hosts all plot docks.
    It is normally created indirectly through :func:`pynaviz.scope`, but can
    also be instantiated directly when embedding pynaviz inside a larger Qt
    application or when writing tests.

    Layout
    ------
    The window is divided into three areas:

    - **Left panel** — ``VariableDock``: a tree widget listing all variables
      passed at construction time.  Double-clicking an entry opens a new plot
      dock for that variable.
    - **Central / right area** — plot docks, one per visualised variable.
      Docks can be dragged, tabbed, floated, or closed.
    - **Bottom status bar** — a global time display and time-unit selector
      that reflects the current playback position shared across all docks.

    Time synchronisation
    --------------------
    All open plot docks share a single :class:`ControllerGroup`
    (``self.ctrl_group``).  Panning or scrubbing in any dock updates the
    shared time and propagates to every other dock automatically.

    Keyboard shortcuts
    ------------------
    Global (window-level):

    - **Space** — play / pause
    - **Ctrl+S** — save layout
    - **Ctrl+O** — load layout

    Per-dock (active when the mouse is over or the canvas has focus):

    - **r** — reset view
    - **← / →** — pan left / right by one page
    - **y** — toggle y-axis lock (span mode only; no-op in x-vs-y or image mode)
    - **x** — toggle x-axis lock (span and image mode; no-op in x-vs-y mode)
    - **Ctrl+← / Ctrl+→** — jump to previous / next superposed epoch; requires at
      least one ``IntervalSet`` to be overlaid on that dock via *Select IntervalSet*
      (works on any plot type, not only ``IntervalSet`` plots)
    - **i / d** — increase / decrease contrast (TsdFrame) or marker size (TsGroup)

    Parameters
    ----------
    variables : dict or None, optional
        Mapping of ``{name: object}`` to populate the variable panel.
        Accepts the same types as :func:`pynaviz.scope`.  Defaults to an
        empty dict (no variables pre-loaded).
    layout_path : str, pathlib.Path, or None, optional
        Path to a ``.json`` layout file produced by *Save Layout*.  When
        given, the window restores the saved dock arrangement, camera views,
        and plot actions immediately after construction.  Variables are
        matched to saved docks by name; unmatched docks are skipped.

    Attributes
    ----------
    variables : dict
        Live mapping of all variables currently known to the window,
        including any loaded via *File → Open* after construction.
    ctrl_group : ControllerGroup
        Shared time controller that synchronises all open plot docks.
    variable_dock : VariableDock
        The left-panel tree widget.
    """

    _file_extensions = {
            "Pynapple": [".npz"],
            "NWB": [".nwb"],
            "Video": [".avi", ".mp4", ".mkv"],
            # uncomment above when PlotAudio is available.
            # "Audio": [".mp3", ".wav", ".flac"]
        }

    def __init__(self, variables: dict | None = None, layout_path: str | pathlib.Path | None = None):
        """
        Raises
        ------
        RuntimeError
            If no ``QApplication`` instance exists.  Create one before
            instantiating ``MainWindow`` directly, or use :func:`pynaviz.scope`
            which handles this automatically.
        """
        if not QApplication.instance():  # pragma: no cover
            raise RuntimeError("A Qt application must be created.")
        super().__init__()

        self.name = "Pynaviz"
        self.setWindowTitle(self.name)
        self.setObjectName(self.name)
        self.resize(QSize(1200, 800))
        self.setDockNestingEnabled(True)
        self._open_file_paths = set()
        self._n_dock_open = 0
        self.variables = variables if variables is not None else {}

        # --- List of variables ---
        self._tsdframe_keys = []
        for k in self.variables.keys():
            if k != "data":
                if isinstance(self.variables[k], nap.TsdFrame):
                    self._tsdframe_keys.append(k)  # Storing tsdframe keys for point and skeleton overlay

        # ---- Top Menu Bar ----
        self._create_top_menu_bar()

        # ---- Variables Widget ----
        self.variable_dock = VariableDock(self.variables, self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.variable_dock)

        # ---- Bottom Status Bar ----
        self._create_status_bar()

        # ---- misc ----
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

    # ------------------------------------------------------------------
    # Menu bar & status bar
    # ------------------------------------------------------------------

    def _create_top_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("&Open...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        open_folder_action = QAction("Open &Folder...", self)
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        file_menu.addSeparator()
        file_menu.addAction("&Load layout", self._load_layout)
        file_menu.addAction("&Save layout", self._save_layout)
        file_menu.addSeparator()
        file_menu.addAction("&Exit", self.close)

        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction("&Shortcuts", self._toggle_help_box)
        help_menu.addAction("&About")
        self.help_box = None

    def _create_status_bar(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)

        # --- play/pause buttons ---
        self.skipBackwardBtn = QPushButton()
        self.skipBackwardBtn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward)
        )
        self.skipBackwardBtn.clicked.connect(self._skip_backward)
        bottom_layout.addWidget(self.skipBackwardBtn)

        self.playPauseBtn = QPushButton()
        self.playPauseBtn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.playPauseBtn.setCheckable(True)
        self.playPauseBtn.toggled.connect(self._toggle_play)
        bottom_layout.addWidget(self.playPauseBtn)

        self.stopBtn = QPushButton()
        self.stopBtn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self.stopBtn.clicked.connect(self._stop)
        bottom_layout.addWidget(self.stopBtn)

        self.skipForwardBtn = QPushButton()
        self.skipForwardBtn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward)
        )
        self.skipForwardBtn.clicked.connect(self._skip_forward)
        bottom_layout.addWidget(self.skipForwardBtn)

        # --- Time spinbox & unit selector ---
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.time_spin_box = QDoubleSpinBox()
        self.time_spin_box.setStyleSheet("font-size: 10pt;")
        self.time_spin_box.setMinimum(0)
        self.time_spin_box.setMaximum(1)
        self.time_spin_box.setValue(0.5)
        self.time_spin_box.valueChanged.connect(self._on_spinbox_changed)
        bottom_layout.addWidget(self.time_spin_box)

        self.time_unit_combo = QComboBox()
        self.time_unit_combo.setStyleSheet("font-size: 10pt;")
        self.time_unit_combo.addItem('us', 1e6)
        self.time_unit_combo.addItem('ms', 1e3)
        self.time_unit_combo.addItem('s', 1.0)
        self.time_unit_combo.setCurrentIndex(2)  # default to seconds
        self.time_unit_combo.setFixedWidth(55)
        self.time_unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        bottom_layout.addWidget(self.time_unit_combo)

        status_bar.addWidget(bottom_container)

    def _toggle_help_box(self):
        from .variable_dock import HelpBox
        # If the box exists and is visible, close it
        if self.help_box and self.help_box.isVisible():
            self.help_box.close()
            return

        self.help_box = HelpBox(parent=self)

        # Position it below the button
        btn_pos = self.mapToGlobal(QPoint(0, 0))
        self.help_box.move(btn_pos)
        self.help_box.show()

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

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

    def open_folder(self):
        """Open a directory containing electrophysiology recordings (e.g. NeuroScopeIO)."""
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self._load_multiple_files([folder])

    def _load_multiple_files(self, filenames: list[str]):
        from .variable_loader import EPHYS_EXTENSIONS

        def get_type(name: pathlib.Path) -> None | Literal["Pynapple", "NWB", "Video", "Ephys"]:
            if name.is_dir():
                return "Ephys"
            file_type = None
            for tp, exts in self._file_extensions.items():
                if name.suffix in exts:
                    file_type = tp
                    break
            if file_type is None and name.suffix.lower() in EPHYS_EXTENSIONS:
                file_type = "Ephys"
            return file_type

        new_vars = {}
        for name in filenames:
            name = pathlib.Path(name)
            file_type = get_type(name)

            if name.name in self.variables:
                continue

            if not name.exists():
                print(f"Path {name} does not exist. Skipping.")
                continue
            elif file_type is None:
                print(f"File type {pathlib.Path(name).suffix} not supported. Skipping.")
                continue
            elif file_type in ["Pynapple"]:
                data = nap.load_file(name)
                if "pynapple" in data.__module__:
                    new_vars.update({name.name: nap.load_file(name)})
                else:
                    print(f"File {name} does not contain a pynapple object. See pynapple documentation for saving pynapple objects with npz")
                    continue
            elif file_type in ["NWB"]:
                data: nap.NWBFile = nap.load_file(name)
                nap_obj_dict = {}
                for key in data.keys():
                    nap_obj_dict[key] = NWBReference(nwb_file=data, key=key)
                new_vars.update({name.name: nap_obj_dict})
            elif file_type == "Video":
                new_vars.update({name.name: name})
            elif file_type == "Ephys":
                try:
                    data = nap.EphysReader(str(name))
                    nap_obj_dict = {key: EphysReference(ephys_reader=data, key=key) for key in data.keys()}
                    new_vars.update({name.name: nap_obj_dict})
                except Exception as e:
                    print(f"Could not load {name} as EphysReader: {e}")
                    continue
            else:
                raise TypeError(f"Developer forgot to add file type `{file_type}` to the loader.")
            self._open_file_paths.add(name.as_posix())
        self.variables.update(new_vars)
        self.variable_dock._add_items_to_tree_widget(new_vars)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            # TODO: look through docks and set a flag for all video plots to use async reading
            self.timer.start(25)  # 40 FPS
            self.playPauseBtn.setIcon(QIcon.fromTheme("media-playback-pause"))
        else:
            # TODO: switch back to normal reading for plotvideos
            self.timer.stop()
            self.playPauseBtn.setIcon(QIcon.fromTheme("media-playback-start"))
            self.ctrl_group.set_interval(self.ctrl_group.current_time, None)

    def _play(self, delta=0.025):
        self.ctrl_group.advance(delta=delta)
        self._update_time_label(self.ctrl_group.current_time)

    def _stop(self):
        if self.playing:
            self._toggle_play()
        self.ctrl_group.set_interval(0, 1)
        self._update_time_label(self.ctrl_group.current_time)

    def _get_max_min_time(self) -> tuple[float, float]:
        max_time = -float("inf")
        min_time = float("inf")
        for dock_widget in self.findChildren(QDockWidget):
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

    def _on_unit_changed(self):
        """When user changes units, update the spinbox display."""
        multiplier = self.time_unit_combo.currentData()
        display_value = self.ctrl_group.current_time * multiplier
        self.time_spin_box.blockSignals(True)  # Prevent recursion
        self.time_spin_box.setValue(display_value)
        self.time_spin_box.blockSignals(False)

    def _on_spinbox_changed(self, value: float):
        """Handle spinbox changes based on source enum."""
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

    # ------------------------------------------------------------------
    # Dock management
    # ------------------------------------------------------------------

    def _cleanup_and_close_dock(self, dock):
        """Properly clean up and close a dock widget."""
        widget = dock.widget()
        if hasattr(widget, 'plot'):
            ctrl_id = widget.plot.controller._controller_id
            widget.close()
            self.ctrl_group.remove(ctrl_id)
        dock.deleteLater()

    def _add_dock_to_gui(self, dock: QDockWidget) -> None:
        """Add dock to the GUI and balance right docks vertically."""
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # Balance heights of all right docks
        right_docks = [
            d for d in self.findChildren(QDockWidget)
            if self.dockWidgetArea(d) == Qt.DockWidgetArea.RightDockWidgetArea
        ]
        if right_docks:
            sizes_h = [1] * len(right_docks)
            self.resizeDocks(right_docks, sizes_h, Qt.Orientation.Vertical)

    def _register_controller(self, widget: object) -> None:
        """Register the widget's plot in the controller group."""
        self.ctrl_group.add(widget.plot, self._n_dock_open)

    def _get_same_level_interval_sets(self, key_path: list[str]) -> dict[str, nap.IntervalSet]:
        """Return all IntervalSets that are siblings of key_path in the variables tree."""
        parent_path = key_path[:-1]
        parent_dict = _get_variable_from_key_path(self.variables, parent_path) if parent_path else self.variables
        if not isinstance(parent_dict, dict):
            return {}
        result = {}
        for sibling_key, sibling_val in parent_dict.items():
            if sibling_key == key_path[-1]:
                continue
            resolved = sibling_val
            if isinstance(resolved, EphysReference):
                resolved = resolved.ephys_reader[resolved.key]
            elif isinstance(resolved, NWBReference):
                resolved = resolved.nwb_file[resolved.key]
            if isinstance(resolved, nap.IntervalSet):
                label = '/'.join(parent_path + [sibling_key])
                result[label] = resolved
        return result

    def _create_widget_for_variable(self, var, key_path: list[str] | None = None) -> object | None:
        """Return the correct widget based on the variable type."""
        index = self._n_dock_open
        if key_path is not None:
            interval_sets = self._get_same_level_interval_sets(key_path)
        else:
            interval_sets = {'/'.join(k): _get_variable_from_key_path(self.variables, k) for k in self.variable_dock._interval_set_key_paths}
        if isinstance(var, nap.TsGroup):
            return TsGroupWidget(var, index=index, set_parent=True, interval_sets=interval_sets)
        elif isinstance(var, nap.Tsd):
            return TsdWidget(var, index=index, set_parent=True, interval_sets=interval_sets)
        elif isinstance(var, nap.TsdFrame):
            return TsdFrameWidget(var, index=index, set_parent=True, interval_sets=interval_sets)
        elif isinstance(var, nap.TsdTensor):
            tsdframes = {k: self.variables[k] for k in self._tsdframe_keys if self.variables[k].shape[1] % 2 == 0}
            return TsdTensorWidget(var, index=index, set_parent=True, tsdframes=tsdframes)
        elif isinstance(var, nap.Ts):
            return TsWidget(var, index=index, set_parent=True)
        elif isinstance(var, nap.IntervalSet):
            return IntervalSetWidget(var, index=index, set_parent=True)
        elif isinstance(var, VideoHandler):
            tsdframes = {k: self.variables[k] for k in self._tsdframe_keys if self.variables[k].shape[1] % 2 == 0}
            return VideoWidget(var, index=index, set_parent=True, tsdframes=tsdframes)
        elif isinstance(var, (str, pathlib.Path)):
            try:
                tsdframes = {k: self.variables[k] for k in self._tsdframe_keys if self.variables[k].shape[1] % 2 == 0}
                return VideoWidget(var, index=index, set_parent=True, tsdframes=tsdframes)
            except Exception as e:
                print(f"Error loading video from '{var}': {e}")
                return None
        elif isinstance(var, NWBReference):
            var = var.nwb_file[var.key]
            return self._create_widget_for_variable(var, key_path=key_path)
        elif isinstance(var, EphysReference):
            var = var.ephys_reader[var.key]
            return self._create_widget_for_variable(var, key_path=key_path)
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

    def add_dock_widget(self, variable: Any, key_path: list[str], state_dict: dict | None = None) -> QDockWidget | None:
        """Add a new dock widget to the main window based on the variable or its key path."""
        widget = self._create_widget_for_variable(variable, key_path=key_path)
        if widget is None:
            return

        # restore manager if any
        if state_dict is not None:
            widget.plot.from_state(state_dict, available_vars=self.variables)

        widget_name = "/".join(key_path)
        dock = self._create_dock(widget_name, widget, key_path)
        self._add_dock_to_gui(dock)
        self._register_controller(widget)
        # This should be incremented only after registering the controller and the dock to the GUI
        self._n_dock_open += 1
        min_time, max_time = self._get_max_min_time()
        if max_time != -float("inf") and min_time != float("inf"):
            time_multiplier = self.time_unit_combo.currentData()
            self.time_spin_box.setMinimum(min_time * time_multiplier)
            self.time_spin_box.setMaximum(max_time * time_multiplier)

        return dock

    def closeEvent(self, event: QEvent):
        """Handle the close event to ensure proper cleanup."""
        for dock in self.findChildren(QDockWidget):
            if dock.objectName() != "VariablesDock":
                self._cleanup_and_close_dock(dock)
        super().closeEvent(event)


def scope(variables: Union[dict, list, tuple, str], layout_path: str = None, ephys_format: str | None = None):
    """Launch the pynaviz GUI and block until the window is closed.

    Parameters
    ----------
    variables : dict, list, tuple, or str
        The data to visualise.  Several input formats are accepted:

        **dict** (recommended) — keys become the display names shown in the
        variable panel; values are the objects to visualise::

            viz.scope({
                "spikes": tsgroup,
                "lfp": tsdframe,
                "epochs": interval_set,
                "recording": "/path/to/video.mp4",
            })

        **list / tuple** — names are inferred automatically from each object's
        class name (``TsGroup``, ``TsdFrame``, …).  Duplicate class names get a
        numeric suffix (``TsGroup_0``, ``TsGroup_1``, …)::

            viz.scope([tsgroup, tsdframe, interval_set])

        **str** — a single file path, treated the same as a one-element list.

        **Supported object types:**

        - ``nap.Ts`` — spike-train / event timestamps
        - ``nap.Tsd`` — one-dimensional time series
        - ``nap.TsdFrame`` — multi-channel time series (pandas DataFrame with time index)
        - ``nap.TsdTensor`` — N-D time series (e.g. pose estimates)
        - ``nap.TsGroup`` — collection of spike trains, optionally with metadata
        - ``nap.IntervalSet`` — epoch / interval data, optionally with metadata
        - ``nap.NWBFile`` — NWB file opened with pynapple; all contained objects
          are unpacked and added individually
        - ``nap.EphysReader`` — Neo-backed electrophysiology reader; all contained
          objects are unpacked and added individually
        - ``str`` / ``pathlib.Path`` pointing to:
            - ``.nwb`` file — loaded via pynapple, objects unpacked as above
            - ``.npz`` file — loaded via pynapple, must contain a single pynapple object
            - ``.mp4`` / ``.avi`` / ``.mov`` / ``.mkv`` — video file, displayed
              in a video player dock
        - ``VideoHandler`` — a pynaviz video handler instance (advanced use;
          allows sharing a pre-initialised decoder across docks)

        Objects that do not match any of the above are silently ignored.

    layout_path : str or None, optional
        Path to a previously saved ``.json`` layout file.  When provided the
        GUI restores the dock arrangement, camera views, and applied actions
        (group-by, sort-by, color-by, interval overlays, …) from that file.
        Variables in *variables* are matched to saved docks by their key name;
        docks whose variable is not found are skipped.
    ephys_format : str or None, optional
        Neo IO class name to use when loading electrophysiology files or
        directories via ``nap.EphysReader`` (e.g. ``"PlexonIO"``,
        ``"NeuroScopeIO"``).  When ``None`` (default) the format is
        auto-detected from the file/directory contents.

    Notes
    -----
    This call blocks until the GUI window is closed.  It starts (or reuses) a
    ``QApplication`` internally, so it is safe to call from a plain Python
    script or a Jupyter notebook.
    """
    variables = get_pynapple_variables(variables, ephys_format=ephys_format)

    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    icon_data = QByteArray.fromBase64(icon_base64)
    pixmap = QPixmap()
    pixmap.loadFromData(icon_data)
    icon = QIcon(pixmap)
    app.setWindowIcon(icon)
    app.setApplicationName("Pynaviz")
    app.setOrganizationName("pynapple-org")
    app.setOrganizationDomain("pynaviz.github.io")

    gui = MainWindow(variables=variables, layout_path=layout_path)

    gui.show()

    app.exit(app.exec())

    gui.close()

    return
