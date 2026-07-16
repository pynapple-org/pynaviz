import warnings

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from pynaviz.qt.interval_sets_selection import ComboDelegate
from pynaviz.skeleton_geometry import (
    edges_from_parent_metadata,
    resolve_edges,
    resolve_keypoint_labels,
)
from pynaviz.utils import GRADED_COLOR_LIST


def _color_icon(name: str) -> QIcon:
    from PySide6.QtGui import QColor, QPixmap

    px = QPixmap(32, 16)
    px.fill(QColor(name))
    return QIcon(px)


class DoubleSpinDelegate(QStyledItemDelegate):

    def __init__(self, min_, max_, parent=None):
        super().__init__(parent)
        self.min_ = min_
        self.max_ = max_

    def createEditor(self, parent, option, index):
        spin = QDoubleSpinBox(parent)
        # Very wide range to simulate "no boundaries"
        spin.setMinimum(self.min_)
        spin.setMaximum(self.max_)
        spin.setSingleStep(1)
        spin.setDecimals(2)  # adjust precision as needed

        # Emit valueChanged signal for convenience (no need any extra signal)
        spin.valueChanged.connect(
            lambda val, ix=index: index.model().setData(ix, val, Qt.ItemDataRole.EditRole)
        )
        return spin

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        if value is None:
            value = 0.0
        editor.setValue(float(value))

    def setModelData(self, editor, model, index):
        editor.interpretText()  # ensure text is parsed
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)


class TsdFramesModel(QAbstractTableModel):
    """A model to handle the dict of tsdframes with checkboxes.

    Used by the video "Overlay points" dialog: points only, so there is no
    line-thickness column (bones are configured via "Overlay Skeleton").
    """

    checkStateChanged = Signal(str, str, float, bool)

    def __init__(self, tsdframes: dict):
        super().__init__()
        self.tsdframes = tsdframes
        self.colors = GRADED_COLOR_LIST
        self.rows = [
            {
                "name": k,
                "colors": self.colors[i%len(self.colors)],
                "markersize": 10,
                "checked": False
            }
            for i, k in enumerate(self.tsdframes.keys())
        ]

    # ---- model dimensions ----
    def rowCount(self, parent=None):
        if parent is None:
            parent = QModelIndex()
        return len(self.rows)

    def columnCount(self, parent=None):
        return 3

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return ["TsdFrame", "Color", "Size"][section]

    def data(self, index, role=None):
        """What to display in the table view."""
        # Guard clause for invalid index
        # (for example, if initialize with empty tsdframe dict)
        if not index.isValid():
            return None

        row, col = index.row(), index.column()
        r = self.rows[row]

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if col == 0:
                return r["name"]
            if col == 1:
                return r["colors"]
            if col == 2:
                return r["markersize"]


        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            return Qt.CheckState.Checked if r["checked"] else Qt.CheckState.Unchecked

        return None

    def flags(self, index):
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            return base | Qt.ItemFlag.ItemIsUserCheckable
        elif index.column() == 1:
            return base | Qt.ItemFlag.ItemIsEditable
        elif index.column() == 2:
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index, value, role=None):
        """
        Write data to the model.

        Parameters
        ----------
        index : QModelIndex
            The index of the item to modify.
        value : Any
            The new value to set.
        role : Qt.ItemDataRole
            The role of the data to set.
        """
        if not index.isValid():
            return False
        row, col = index.row(), index.column()
        r = self.rows[row]

        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            # handles both  Qt.CheckState Enum and int/bool
            check_value = getattr(value, 'value', value)
            r["checked"] = (check_value == Qt.CheckState.Checked.value)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            self.checkStateChanged.emit(r["name"], r["colors"], r["markersize"], r["checked"])
            return True

        if role == Qt.ItemDataRole.EditRole:
            if col == 1:
                r["colors"] = str(value)
            elif col == 2:
                r["markersize"] = float(value)
            else:
                return False
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.EditRole])
            self.checkStateChanged.emit(r["name"], r["colors"], r["markersize"], r["checked"])
            return True
        return False

class TsdFramesDialog(QDialog):
    """
    Dialog showing a table of tsdframe with 3 columns:
    - Column 0: name + checkbox
    - Column 1: dropdown combo (color)
    - Column 2: number entry (marker size)

    Points only; use "Overlay Skeleton" to draw connecting bones.
    """
    def __init__(self, model: TsdFramesModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("TsdFrame selection")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumSize(400, 300)

        self.view = QTableView(self)
        self.view.setModel(model)
        header = self.view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # header.setStretchLastSection(True)
        # self.view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)


        color_delegate = ComboDelegate(self.view)
        self.view.setItemDelegateForColumn(1, color_delegate)

        # Marker size
        markersize_delegate = DoubleSpinDelegate(min_=0, max_=1e12, parent=self.view)
        self.view.setItemDelegateForColumn(2, markersize_delegate)

        layout = QVBoxLayout()

        # Add a help message
        text = ("Select the TsdFrame to overlay as points. \n"
                "Adjust color and marker size as needed. \n"
                "TsdFrame object should have even number of columns representing x,y coordinates. \n"
                "Ex : (x1, y1, x2, y2, ...) \n"
                "Use \"Overlay Skeleton\" to draw connecting bones. \n")
        help_label = QLabel(text)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        layout.addWidget(self.view)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.setDefault(True)
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.adjustSize()


def _effective_edge_names(labels, edges, data) -> list:
    """Current skeleton bones as ``(labelA, labelB)`` name pairs.

    Mirrors what would actually be drawn, so the editor opens seeded with the
    visible skeleton: explicit ``edges`` if set, else the data's ``"parent"``
    metadata, else the complete-graph fallback.
    """
    labels = list(labels)
    if edges is None:
        edges = edges_from_parent_metadata(data)
    if edges is None:
        with warnings.catch_warnings():
            # The complete-graph fallback warns; seeding the editor isn't the
            # place to surface that (the user is about to pick real bones).
            warnings.simplefilter("ignore")
            idx_pairs = resolve_edges(labels, None)
        return [(labels[i], labels[j]) for i, j in idx_pairs]
    # Normalize any index entries to labels; leave name entries as-is.
    return [
        (a if isinstance(a, str) else labels[int(a)],
         b if isinstance(b, str) else labels[int(b)])
        for a, b in edges
    ]


def _remove_video_overlay(plot, label) -> None:
    """Remove a superposed points/skeleton overlay from a video plot."""
    pts = plot.points.pop(label, None)
    if pts is not None:
        pts._geometry.remove_from_scene(plot.scene)


class _PlotSkeletonController:
    """SkeletonDialog backend for a ``PlotTsdFrame`` (its single skeleton mode)."""

    title = "Plot Skeleton"

    def __init__(self, plot):
        self.plot = plot
        self.mode = plot._modes["skeleton"]

    def source_names(self):
        return []  # single source -> no TsdFrame dropdown

    def has_show_toggle(self):
        return False

    def select(self, name=None):
        m = self.mode
        return {
            "labels": list(m.labels),
            "edges": _effective_edge_names(m.labels, m.edges, m.data),
            "color": m.color,
            "thickness": m.thickness,
            "markersize": m.markersize,
            "shown": True,
        }

    def apply(self, edges, color, thickness, markersize):
        self.plot.plot_skeleton(
            edges=edges, color=color, thickness=thickness, markersize=markersize
        )

    def set_shown(self, shown, edges, color, thickness, markersize):
        pass  # always shown

    def snapshot(self):
        return self.plot.get_plot_state()

    def restore(self, snapshot):
        self.plot.set_plot_state(snapshot)


class _VideoSkeletonController:
    """SkeletonDialog backend for video overlays: one skeleton per TsdFrame.

    Several even-column TsdFrames can be overlaid, each with its own bones and
    styling. A dropdown picks the active TsdFrame; per-frame params and show
    state are remembered so switching back restores what was set.
    """

    title = "Overlay Skeleton"

    def __init__(self, plot, tsdframes):
        self.plot = plot
        self.tsdframes = {k: v for k, v in tsdframes.items() if v.shape[1] % 2 == 0}
        self._current = None
        self._cache = {}   # name -> {edges, color, thickness, markersize}
        self._shown = {}   # name -> bool

    def source_names(self):
        return list(self.tsdframes)

    def has_show_toggle(self):
        return True

    def select(self, name):
        self._current = name
        tsdframe = self.tsdframes[name]
        labels = resolve_keypoint_labels(tsdframe.columns)
        if name not in self._cache:
            self._cache[name] = {
                "edges": _effective_edge_names(labels, None, tsdframe),
                "color": GRADED_COLOR_LIST[len(self._cache) % len(GRADED_COLOR_LIST)],
                "thickness": 0.02,
                "markersize": 10.0,
            }
        return {
            "labels": labels,
            "shown": self._shown.get(name, True),
            **self._cache[name],
        }

    def apply(self, edges, color, thickness, markersize):
        if self._current is None:
            return
        self._cache[self._current] = {
            "edges": edges, "color": color,
            "thickness": thickness, "markersize": markersize,
        }
        if self._shown.get(self._current, True):
            self._overlay(self._current)

    def set_shown(self, shown, edges, color, thickness, markersize):
        if self._current is None:
            return
        self._shown[self._current] = shown
        self._cache[self._current] = {
            "edges": edges, "color": color,
            "thickness": thickness, "markersize": markersize,
        }
        if shown:
            self._overlay(self._current)
        else:
            _remove_video_overlay(self.plot, self._current)
            self.plot.canvas.request_draw(self.plot.animate)

    def _overlay(self, name):
        p = self._cache[name]
        _remove_video_overlay(self.plot, name)
        self.plot.superpose_points(
            self.tsdframes[name], color=p["color"], markersize=p["markersize"],
            thickness=p["thickness"], edges=p["edges"], label=name,
        )

    def snapshot(self):
        return self.plot.get_plot_state()

    def restore(self, snapshot):
        for label in list(self.plot.points):
            _remove_video_overlay(self.plot, label)
        self.plot.set_plot_state(snapshot, self.tsdframes)
        self.plot.canvas.request_draw(self.plot.animate)


class SkeletonDialog(QDialog):
    """Configure a skeleton (bones + point/line styling) with live preview.

    Backed by a controller so the same editor drives both a ``PlotTsdFrame``
    skeleton and video-overlay skeletons — build one with :meth:`for_plot` or
    :meth:`for_video`. For the video case a TsdFrame dropdown and a "Show this
    skeleton" checkbox appear at the top. Changes preview live; OK keeps them,
    Cancel reverts.
    """

    @classmethod
    def for_plot(cls, plot, parent: QWidget | None = None):
        """Dialog editing a ``PlotTsdFrame``'s skeleton mode."""
        return cls(_PlotSkeletonController(plot), parent)

    @classmethod
    def for_video(cls, plot, tsdframes, parent: QWidget | None = None):
        """Dialog editing skeleton overlays on a video plot."""
        return cls(_VideoSkeletonController(plot, tsdframes), parent)

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller
        self.labels = []

        # Changes apply live; only turn that on once the widgets are built and
        # seeded, so seeding doesn't fire updates. `_loading` guards reseeding
        # when the active TsdFrame changes.
        self._live_enabled = False
        self._loading = False
        # Snapshot to restore on Cancel (reverts the live edits).
        self._snapshot = controller.snapshot()

        self.setWindowTitle(getattr(controller, "title", "Plot Skeleton"))
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumSize(620, 560)

        layout = QVBoxLayout(self)

        help_label = QLabel(
            "Define the skeleton bones as pairs of keypoints, and adjust the "
            "point color, marker size and line thickness.\n"
            "Changes preview live; OK keeps them, Cancel reverts.\n"
            "An empty bone list draws the keypoints only (no connecting lines)."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        # ---- source selector (video overlay: pick which TsdFrame) ----
        self.source_combo = None
        self.show_check = None
        names = controller.source_names()
        if names or controller.has_show_toggle():
            source_row = QHBoxLayout()
            if names:
                self.source_combo = QComboBox()
                self.source_combo.addItems(names)
                source_row.addWidget(QLabel("TsdFrame"))
                source_row.addWidget(self.source_combo, 1)
            if controller.has_show_toggle():
                self.show_check = QCheckBox("Show this skeleton")
                source_row.addWidget(self.show_check)
            source_row.addStretch()
            layout.addLayout(source_row)

        # ---- styling controls (single row); values seeded in _load_source ----
        self.color_combo = QComboBox()
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setDecimals(1)
        self.size_spin.setRange(0.0, 1e3)
        self.size_spin.setSingleStep(1.0)
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setDecimals(3)
        self.thickness_spin.setRange(0.0, 1e3)
        self.thickness_spin.setSingleStep(0.01)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Color"))
        style_row.addWidget(self.color_combo, 1)
        style_row.addSpacing(12)
        style_row.addWidget(QLabel("Marker size"))
        style_row.addWidget(self.size_spin)
        style_row.addSpacing(12)
        style_row.addWidget(QLabel("Line thickness"))
        style_row.addWidget(self.thickness_spin)
        layout.addLayout(style_row)

        # ---- bone table ----
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Keypoint A", "Keypoint B"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        edit_buttons = QHBoxLayout()
        add_button = QPushButton("Add bone")
        remove_button = QPushButton("Remove selected")
        add_button.clicked.connect(self._on_add_row)
        remove_button.clicked.connect(self._on_remove_rows)
        edit_buttons.addWidget(add_button)
        edit_buttons.addWidget(remove_button)
        edit_buttons.addStretch()
        layout.addLayout(edit_buttons)

        # ---- metadata info ----
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        info_label = QLabel(
            "Bones can also be stored on the data as <code>parent</code> metadata, "
            "so they load automatically. Use one value per column (a keypoint's "
            "<code>_x</code> and <code>_y</code> share it) naming the keypoint it "
            "connects to, or <code>None</code> for the root. Clear the list above "
            "to fall back to this metadata.<br><br>"
            "Example — <code>tsdframe.set_info(parent=[...])</code>:"
            "<pre>columns  Back_x Back_y  Nose_x Nose_y  Tail_x Tail_y\n"
            "parent   None   None    'Back' 'Back'  'Back' 'Back'</pre>"
            "&rarr; bones: Nose&ndash;Back, Tail&ndash;Back"
        )
        info_label.setTextFormat(Qt.TextFormat.RichText)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        # ---- OK / Cancel ----
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.setDefault(True)
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)

        self.resize(680, 620)

        # Seed the widgets from the initial source, then wire signals and go
        # live so the seeded skeleton previews immediately.
        self._load_source(names[0] if names else None)
        self.color_combo.currentIndexChanged.connect(self._on_change)
        self.size_spin.valueChanged.connect(self._on_change)
        self.thickness_spin.valueChanged.connect(self._on_change)
        if self.source_combo is not None:
            self.source_combo.currentTextChanged.connect(self._on_source_changed)
        if self.show_check is not None:
            self.show_check.toggled.connect(self._on_show_toggled)

        self._live_enabled = True
        self._apply_live()

    def _load_source(self, name) -> None:
        """Seed all widgets from the controller's state for ``name``."""
        self._loading = True
        try:
            state = self.controller.select(name)
            self.labels = list(state["labels"])

            self.color_combo.clear()
            items = list(GRADED_COLOR_LIST)
            color = state["color"]
            # A color outside the graded list (e.g. an rgba tuple or a contrast
            # default) is surfaced so the combo reflects the real current value.
            if isinstance(color, str) and color not in items:
                items.insert(0, color)
            for n in items:
                self.color_combo.addItem(_color_icon(n), n)
            if isinstance(color, str) and color in items:
                self.color_combo.setCurrentIndex(items.index(color))

            self.size_spin.setValue(float(state["markersize"]))
            self.thickness_spin.setValue(float(state["thickness"]))

            self.table.setRowCount(0)
            for a, b in state["edges"]:
                self._add_edge_row(a, b)

            if self.show_check is not None:
                self.show_check.setChecked(bool(state["shown"]))
        finally:
            self._loading = False

    def _on_source_changed(self, name) -> None:
        """Switch the active TsdFrame: reseed the editor and preview it."""
        self._load_source(name)
        self._on_change()

    def _on_show_toggled(self, checked) -> None:
        """Toggle whether the current TsdFrame's skeleton is overlaid."""
        if self._loading:
            return
        self.controller.set_shown(
            checked, self.edges(), self.color_combo.currentText(),
            self.thickness_spin.value(), self.size_spin.value(),
        )

    def _on_change(self, *args) -> None:
        """Live-apply the current dialog state (unless loading/not yet live)."""
        if self._loading or not self._live_enabled:
            return
        self._apply_live()

    def _apply_live(self) -> None:
        # A hidden skeleton (video overlay with "Show" unchecked) isn't drawn.
        if self.show_check is not None and not self.show_check.isChecked():
            return
        self.controller.apply(
            self.edges(),
            self.color_combo.currentText(),
            self.thickness_spin.value(),
            self.size_spin.value(),
        )

    def _make_keypoint_combo(self, selected: str | None = None) -> QComboBox:
        combo = QComboBox()
        combo.addItems(self.labels)
        if selected is not None and selected in self.labels:
            combo.setCurrentIndex(self.labels.index(selected))
        combo.currentIndexChanged.connect(self._on_change)
        return combo

    def _add_edge_row(self, a: str | None = None, b: str | None = None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        # A button-added bone defaults to the first two distinct keypoints.
        if a is None and self.labels:
            a = self.labels[0]
        if b is None and len(self.labels) > 1:
            b = self.labels[1]
        self.table.setCellWidget(row, 0, self._make_keypoint_combo(a))
        self.table.setCellWidget(row, 1, self._make_keypoint_combo(b))

    def _on_add_row(self) -> None:
        self._add_edge_row()
        self._on_change()

    def _remove_selected_rows(self) -> None:
        rows = sorted({ix.row() for ix in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _on_remove_rows(self) -> None:
        self._remove_selected_rows()
        self._on_change()

    def edges(self) -> list:
        """Collect the table's bones as ``(labelA, labelB)`` pairs.

        Rows connecting a keypoint to itself are dropped (a self-loop draws no
        bone anyway).
        """
        out = []
        for row in range(self.table.rowCount()):
            a = self.table.cellWidget(row, 0).currentText()
            b = self.table.cellWidget(row, 1).currentText()
            if a and b and a != b:
                out.append((a, b))
        return out

    def reject(self) -> None:
        """Cancel: revert the live edits by restoring the opening state."""
        self._live_enabled = False
        self.controller.restore(self._snapshot)
        super().reject()

