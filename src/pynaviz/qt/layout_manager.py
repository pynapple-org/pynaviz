"""
LayoutManagerMixin provides methods for saving and restoring the MainWindow layout, including the geometry, dock arrangement, and plot states.  It expects to be mixed
into a MainWindow class that has certain attributes (e.g. ``self.variables``, ``self.ctrl_group``, etc.) and provides methods like ``add_dock_widget``.  The layout is saved
to a JSON file, with the geometry and state encoded in base64.  When restoring, it recreates the docks based on the saved variable key paths and restores their views and the global time.
"""

import base64
import json
import os
import pathlib
import re
from datetime import datetime

from PySide6.QtCore import QByteArray
from PySide6.QtWidgets import QDockWidget, QFileDialog

from ..controller import GetController
from .variable_dock import _get_variable_from_key_path


class LayoutManagerMixin:
    """Mixin providing layout save / restore functionality for MainWindow.

    All methods operate on ``self`` (the host ``MainWindow`` instance) and
    expect the following attributes to already exist:

    - ``self.variables`` — live variables dict
    - ``self.ctrl_group`` — shared ControllerGroup
    - ``self._n_dock_open`` — running dock counter
    - ``self._open_file_paths`` — set of file paths loaded via File→Open
    - ``self.variable_dock`` — VariableDock instance
    """

    def _load_layout(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Layout", "", "Layout Files (*.json)")
        if file_name:
            self._restore_layout(file_name)

    def _get_current_dock_name(self, saved_name: str) -> str:
        """Translate a saved dock name to the name it will have after recreation.

        Dock names encode the creation index (e.g. ``tsdframe_3``).  When docks
        are closed and reopened the index restarts from 0, so the stored name
        may not match the newly-created one.  This replaces the trailing integer
        with the current ``_n_dock_open - 1`` counter.
        """
        return re.sub(r"_\d+$", f"_{self._n_dock_open - 1}", saved_name)

    def _get_sorted_plot_docks(self) -> list[QDockWidget]:
        """Return all non-variable dock widgets sorted by their creation index."""
        return sorted(
            [d for d in self.findChildren(QDockWidget) if d.objectName() != "VariablesDock"],
            key=lambda d: int(d.objectName().split("_")[-1]),
        )

    def _clear_layout(self):
        """Remove all plot docks, reset controller group and dock counter."""
        for d in self.findChildren(QDockWidget):
            if d.objectName() != "VariablesDock":
                d.close()
                d.setParent(None)
                d.deleteLater()
        self.restoreState(QByteArray())
        self.restoreGeometry(QByteArray())
        for ctrl_id in list(self.ctrl_group._controller_group.keys()):
            self.ctrl_group.remove(ctrl_id)
        self._n_dock_open = 0
        assert len(self.ctrl_group._controller_group) == 0, "Controller group not empty after removing all docks."

    def _restore_docks(self, docks_payload: list[dict]) -> dict[str, QDockWidget]:
        """Recreate each dock from the payload, skipping missing variables.

        Returns a mapping from the saved dock name to the live QDockWidget after
        recreation.  The index embedded in the name changes when docks are closed
        and reopened, so the saved name cannot be used directly.

        Note: simply matching the widget name stored in the payload results in a bug
        when the following happens:
        1. Open > 1 docks
        2. Close any dock but the last -> now the last dock name is varname_N but N-1 docks are opened
        3. Save layout and close, this will store widget["name"] == varname_N
        4. Try to reopen loading the layout.
           The last dock name will be varname_{N-1} but widget["name"] is varname_N
        To prevent this, adjust the index via ``_get_current_dock_name``.
        """
        saved_to_dock: dict[str, QDockWidget] = {}
        for widget in docks_payload:
            var = _get_variable_from_key_path(self.variables, widget['key_path'])
            if var is None:
                print(f"Variable '{widget['name']}' not found. Skipping dock.")
                continue
            print(f"Adding var from path {widget['key_path']}.")
            self.add_dock_widget(var, widget['key_path'], state_dict=widget["state_dict"])
            dock_name = self._get_current_dock_name(widget['name'])
            dock = self.findChild(QDockWidget, dock_name)
            if dock is None:
                raise RuntimeError(f"Dock {widget['name']} was not created.")
            saved_to_dock[widget['name']] = dock
        return saved_to_dock

    def _restore_views(self, docks_payload: list[dict], saved_to_dock: dict[str, QDockWidget]):
        """Restore camera views and current time for all plot docks.

        Restoration order logic:
        1. First restore the views of SpanControllers. This will sync xlim (and therefore time) and ylim.
        2. Restore the GetController views (this moves the camera but no time change).
        3. Set the global time. This will be a no-op for the SpanController but will change the frames of the
        Get controllers.
        """
        current_time = None
        get_ctrl_views = []
        # Loop over docks and set view only for SpanControllers
        for widget_info in docks_payload:
            dock = saved_to_dock.get(widget_info['name'])
            if dock is None:
                continue
            state_dict = widget_info.get("state_dict", {})
            if current_time is None:
                current_time = state_dict.get("current_time")
            view = state_dict.get("view")
            if view is None:
                continue
            ctrl = dock.widget().plot.controller
            if isinstance(ctrl, GetController):
                get_ctrl_views.append((ctrl, view))
            else:
                ctrl.set_view(*view)
        # Loop over docks with a GetController
        for ctrl, view in get_ctrl_views:
            ctrl.set_view(*view)
        # Set the global time
        if current_time is not None:
            self.ctrl_group.set_interval(start=current_time, end=None)

    def _restore_geometry(self, payload: dict):
        """Restore Qt window geometry and dock layout from the payload."""
        geom = QByteArray.fromBase64(payload["geometry_b64"].encode("ascii"))
        state = QByteArray.fromBase64(payload["state_b64"].encode("ascii"))
        try:
            self.restoreGeometry(geom)
            ok = self.restoreState(state, payload.get("version", 0))
            print("restoreState ok:", ok)
        except Exception as e:
            print("Error restoring layout:", e)

    def _restore_layout(self, file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self._clear_layout()
        self._load_multiple_files(payload.get("file_paths", []))
        docks_payload = payload.get("docks", [])
        saved_to_dock = self._restore_docks(docks_payload)
        self._restore_views(docks_payload, saved_to_dock)
        self._restore_geometry(payload)

    def _get_plot_state(self, plot):
        """Collect plot state, remapping auto-generated labels to variable names.

        When a user calls ``plot.add_interval_sets(epochs)`` or
        ``plot.superpose_points(tsdframe)`` without an explicit label, pynaviz
        auto-generates one (e.g. ``interval_0``, ``points_1``). This method
        replaces such labels with the matching key from ``self.variables`` using
        object identity, so that they can be looked up by name on load.
        Labels that have no matching variable are kept as-is.
        """
        state = plot.get_state()
        id_to_name = {id(v): k for k, v in self.variables.items()}

        # remap IntervalSets and tsdframes
        # (when named with default labels `interval_set_n`)
        remapped_iset = {}
        for label, props in state.get("interval_sets", {}).items():
            epoch = plot._epochs.get(label)
            var_name = id_to_name.get(id(epoch), label)
            remapped_iset[var_name] = props
        state["interval_sets"] = remapped_iset
        if hasattr(plot, "points"):
            remapped_tsdframes = {}
            for label, props in state.get("plot", {}).items():
                tsdframe = plot.points[label].data
                var_name = id_to_name.get(id(tsdframe), label)
                remapped_tsdframes[var_name] = props
            state["plot"] = remapped_tsdframes
        state["current_time"] = self.ctrl_group.current_time
        state["view"] = list(plot.controller.get_view())
        return state

    def _get_layout_dict(self):
        geom = bytes(self.saveGeometry())
        state = bytes(self.saveState(version=0))  # Potentially add package versioning later

        all_docks = self.findChildren(QDockWidget)
        docks = []
        order = []
        for d in all_docks:
            name = d.objectName()
            if name != "VariablesDock":
                info = {"visible": d.isVisible(),
                        "floating": d.isFloating(),
                        "area": self.dockWidgetArea(d).name,
                        "pos": (d.x(), d.y()),
                        "size": (d.width(), d.height()),
                        "dtype": d.widget().plot.data.__class__.__name__,
                        "key_path": d.property("key_path"),
                        "index": int(name.split("_")[-1]),
                        "name": name,
                        "state_dict": self._get_plot_state(d.widget().plot),
                        }
                docks.append(info)
                order.append(int(name.split("_")[-1]))
        docks = [x for _, x in sorted(zip(order, docks))]

        payload = {
            "version": 0,
            "geometry_b64": base64.b64encode(geom).decode("ascii"),
            "state_b64": base64.b64encode(state).decode("ascii"),
            "docks": docks,
            "file_paths": list(self._open_file_paths),  # to reload files on layout load
        }
        return payload

    def _save_layout(self, file_name: str | pathlib.Path | None = None):
        print("Saving layout...")
        if file_name is None:
            dt = datetime.now().strftime("%Y-%m-%d_%H-%M")
            default_file = os.path.join(os.getcwd(), f"layout_{dt}.json")
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Layout",
                default_file,  # suggested file name
                "Layout Files (*.json)"
            )

        if file_name:
            file_name = pathlib.Path(file_name) if isinstance(file_name, str) else file_name
            file_name = file_name.with_suffix(".json")

            payload = self._get_layout_dict()

            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Layout saved to {file_name}")
