"""
Action and context classes

This module provides custom Qt widgets to create interactive menus for plots.
Each menu allows the user to configure plot-specific behavior using GUI components
such as dropdowns, spin boxes, and list views. The widgets are dynamically constructed
based on metadata and plotting context.

Main Classes:
- DropdownDialog: Dynamically generates a dialog with labeled input widgets.
- MenuWidget: UI component to attach interactive actions and selections to a plot.
"""

from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import pynapple as nap
from PyQt6.QtCore import QPoint, QSize, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from pynaviz.qt.drop_down_dict_builder import get_popup_kwargs
from pynaviz.qt.interval_sets_selection import IntervalSetsDialog, IntervalSetsModel
from pynaviz.qt.tsdframe_selection import TsdFramesDialog, TsdFramesModel
from pynaviz.qt.widget_list_selection import (
    ChannelList,
    ChannelListModel,
)
from pynaviz.utils import get_plot_attribute

WIDGET_PARAMS = {
    QComboBox: {
        "name": "setObjectName",
        "items": "addItems",
        "values": "setItemData",
        "current_index": "setCurrentIndex",
    },
    QDoubleSpinBox: {
        "name": "setObjectNAme",  # Note: typo here in key name
        "value": "setValue",
    },
}


def widget_factory(parameters: dict) -> QWidget:
    """
    Constructs a QWidget (QComboBox or QDoubleSpinBox) with specified parameters.

    Parameters
    ----------
    parameters : dict
        Dictionary containing widget configuration.

    Returns
    -------
    QWidget
        The configured widget instance.
    """
    widget_type = parameters.pop("type")
    if widget_type == QComboBox:
        widget = QComboBox()
        for arg_name, attr_name in WIDGET_PARAMS[QComboBox].items():
            method = getattr(widget, attr_name, None)
            value = parameters.get(arg_name)
            if method and value is not None:
                if arg_name == "values":
                    for i, v in enumerate(value):
                        method(i, v)
                else:
                    method(value)
    elif widget_type == QDoubleSpinBox:
        widget = QDoubleSpinBox()
        for arg_name, attr_name in WIDGET_PARAMS[QDoubleSpinBox].items():
            method = getattr(widget, attr_name, None)
            value = parameters.get(arg_name)
            if method and value is not None:
                method(value)
    else:
        raise ValueError("Unknown widget type.")
    return widget


class DropdownDialog(QDialog):
    """
    A popup dialog that dynamically creates widgets from metadata and applies a callback.

    Parameters
    ----------
    title : str
        Title of the dialog window.
    widgets : OrderedDict[str, dict]
        Keys are labels; values are widget parameter dictionaries.
    func : Callable
        Function to call when selections are made.
    ok_cancel_button : bool, optional
        Whether to display OK/Cancel buttons (default is False).
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        title: str,
        widgets: OrderedDict[str, dict],
        func: Callable,
        ok_cancel_button: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.NonModal)

        num_cols = min(len(widgets), 3)
        num_rows = len(widgets) // num_cols
        self.setFixedWidth(200 * num_cols)
        self.setFixedHeight(min(150 * num_rows, 400))

        self._func = func
        self.widgets: dict[int, QWidget] = {}

        main_layout = QVBoxLayout(self)

        # Scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll_content = QWidget()
        scroll_content.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        grid_layout = QGridLayout()
        inner_layout = QVBoxLayout()
        inner_layout.addLayout(grid_layout)

        spacer = QSpacerItem(
            0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )
        h_spacer = QSpacerItem(
            0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        inner_layout.addItem(spacer)

        outer_layout = QHBoxLayout()
        outer_layout.addLayout(inner_layout)
        outer_layout.addItem(h_spacer)

        scroll_content.setLayout(outer_layout)
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # Add widgets with labels
        def make_labeled_widget(label_text: str, widget: QWidget) -> QWidget:
            label = QLabel(label_text)
            label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            wrapper = QWidget()
            layout = QHBoxLayout()
            layout.setContentsMargins(1, 0, 1, 0)
            layout.setSpacing(2)
            layout.addWidget(label)
            layout.addWidget(widget)
            wrapper.setLayout(layout)
            return wrapper

        for i, (label, params) in enumerate(widgets.items()):
            widget = widget_factory(params)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.item_changed)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.item_changed)

            row, col = divmod(i, num_cols)
            grid_layout.addWidget(make_labeled_widget(label, widget), row, col)
            self.widgets[i] = widget

        # Optional buttons
        if ok_cancel_button:
            self._update_on_selection = False
            button_layout = QHBoxLayout()
            self.ok_button = QPushButton("OK")
            self.ok_button.setDefault(True)
            cancel_button = QPushButton("Cancel")

            self.ok_button.clicked.connect(self.accept)
            cancel_button.clicked.connect(self.reject)

            button_layout.addStretch()
            button_layout.addWidget(cancel_button)
            button_layout.addWidget(self.ok_button)
            main_layout.addLayout(button_layout)
        else:
            self._update_on_selection = True

        self.adjustSize()

    def get_selections(self) -> list[Any]:
        """
        Extracts current selections from the widgets.

        Returns
        -------
        list
            List of selected values from each widget.
        """
        out = []
        for widget in self.widgets.values():
            if isinstance(widget, QComboBox):
                data = widget.currentData()
                out.append(data if data is not None else widget.currentText())
            elif isinstance(widget, QDoubleSpinBox):
                out.append(widget.value())
        return out

    def update_plot(self) -> None:
        """Calls the provided function with current widget values."""
        self._func(*self.get_selections())

    def item_changed(self) -> None:
        """Callback triggered when a widget value changes."""
        if self._update_on_selection:
            self.update_plot()

    def accept(self) -> None:
        """Override accept to call plot update before closing."""
        self.update_plot()
        return super().accept()


class MenuWidget(QWidget):
    """
    Menu bar widget that allows all possible actions

    Parameters
    ----------
    metadata : dict or pd.DataFrame or None
        Metadata associated with the plot.
    plot : _BasePlot
        The plot instance this menu is attached to.
    interval_sets : dict, optional
        Dictionary of interval sets that can be added to the plot. Specific to Tsd, TsdFrame and TsGroup plots.
    tsdframes : dict, optional
        TsdFrame object for overlaying on TsdTensor plot or VideoWidget plot.
    """

    def __init__(self, metadata: Any, plot: Any, interval_sets: dict | None = None, tsdframes: dict | None = None):
        super().__init__()
        self._interval_sets = None
        self._interval_sets_model = None
        self._tsdframes = None
        self._tsdframes_model = None
        self.metadata = metadata
        self.plot = plot

        self.channel_model = ChannelListModel(self.plot.data)
        self.channel_model.checkStateChanged.connect(self._change_visibility)
        self.icon_size = 15

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Channel selection available
        if hasattr(plot._data, "metadata"):
            self._add_button_to_layout(
                layout=layout,
                attr_name="select_button",
                callback=self.show_select_menu,
                icon_name="SP_DialogApplyButton",
                icon_size=self.icon_size
            )

        # Action menu for plot operations
        self.action_funcs = {}
        if metadata is not None and hasattr(metadata, "shape") and np.prod(metadata.shape):
            self.action_funcs = {
                "color_by": "Color by",
                "group_by": "Group by",
                "sort_by": "Sort by",
            }

        # IntervalSet selection available
        if interval_sets is not None and isinstance(interval_sets, dict) and len(interval_sets):
            self._set_interval_sets(interval_sets)

        # TsdFrame overlay available
        if tsdframes is not None and isinstance(tsdframes, dict) and len(tsdframes):
            self._set_tsdframes(tsdframes)

        # Add the action button only if there are actions to show
        if len(self.action_funcs):
            self._add_button_to_layout(
                layout=layout,
                attr_name="action_button",
                callback=self.show_action_menu,
                icon_name="SP_FileDialogDetailedView",
                icon_size=self.icon_size
            )

        # Navigation buttons for time-based data
        if isinstance(plot._data, (nap.Ts, nap.IntervalSet)):
            self._add_button_to_layout(
                layout=layout,
                attr_name="left_jump_button",
                callback=self.jump_previous,
                icon_name="SP_ArrowLeft",
                icon_size=self.icon_size
            )
            self._add_button_to_layout(
                layout=layout,
                attr_name="right_jump_button",
                callback=self.jump_next,
                icon_name="SP_ArrowRight",
                icon_size=self.icon_size
            )

        layout.addStretch()
        self.setLayout(layout)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed
        )
        self._action_menu()

    def _set_interval_sets(self, interval_sets: dict) -> None:
        """Set or update the IntervalSet overlays available for selection."""
        if not all(isinstance(v, nap.IntervalSet) for v in interval_sets.values()):
            raise ValueError("All values in interval_sets must be nap.IntervalSet instances.")
        self.action_funcs["select_interval_set"] = "Select IntervalSet"
        self._interval_sets_model = IntervalSetsModel(interval_sets)
        self._interval_sets_model.checkStateChanged.connect(self._set_interval_set)
        self._interval_sets = interval_sets

    def _set_tsdframes(self, tsdframes: dict) -> None:
        """Set or update the TsdFrame overlays available for selection."""
        if not all(isinstance(v, nap.TsdFrame) for v in tsdframes.values()):
            raise ValueError("All values in tsdframes must be nap.TsdFrame instances.")
        self.action_funcs["overlay_time_series"] = "Overlay points"
        self._tsdframes_model = TsdFramesModel(tsdframes)
        self._tsdframes_model.checkStateChanged.connect(self._overlay_time_series)
        self._tsdframes = tsdframes

    def _change_visibility(self) -> None:
        """Request a redraw of the plot when channel states change."""
        widget = self.sender()
        visibility= np.array([val for val in getattr(widget, "checks", []).values()])
        if hasattr(self.plot, "_manager"):
            self.plot._manager.visible = visibility
        if hasattr(self.plot, "_update"):
            self.plot._update("toggle_visibility")

    def _add_button_to_layout(self, layout, attr_name, callback, icon_name, icon_size):
        button = self._make_button(callback, icon_name, icon_size)
        setattr(self, attr_name, button)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)

    def _make_button(
        self, menu_to_show: Callable, icon_name: str, icon_size: int = 20
    ) -> QPushButton:
        """Helper to create a styled button with icon and action."""
        button = QPushButton()
        icon = self.style().standardIcon(getattr(QStyle.StandardPixmap, icon_name))
        button.setIcon(icon)
        button.setIconSize(QSize(icon_size, icon_size))
        button.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Minimum
        )
        button.setFixedSize(icon_size + 4, icon_size + 4)
        button.setFlat(True)
        button.clicked.connect(menu_to_show)
        return button

    def _action_menu(self) -> None:
        """Creates the action menu with plot operation entries."""
        self.action_menu = QMenu()
        for func_name, name in self.action_funcs.items():
            if func_name in ["select_interval_set", "overlay_time_series"]:
                self.action_menu.addSeparator()
            action = self.action_menu.addAction(name)
            action.setObjectName(func_name)
            action.triggered.connect(self._popup_menu)

    def show_action_menu(self) -> None:
        """Displays the action menu below the button."""

        if hasattr(self.plot, "_controllers"):
            # If 'get' controller is enabled (i.e., in x vs y mode),
            # Need to disable all others actions
            if self.plot._controllers["get"].enabled:
                for act in self.action_menu.actions():
                    act.setEnabled(act.objectName() == "x_vs_y")
            else:
                for act in self.action_menu.actions():
                    act.setEnabled(True)

        pos = self.action_button.mapToGlobal(QPoint(0, self.action_button.height()))
        self.action_menu.exec(pos)

    def show_select_menu(self) -> None:
        """Opens the channel list selection dialog."""
        if hasattr(self.plot, "_controllers"):
            # If 'get' controller is enabled (i.e., in x vs y mode),
            # Need to disable channel selection
            if self.plot._controllers["get"].enabled:
                return
        dialog = ChannelList(self.channel_model, parent=self)
        dialog.show()

    def show_overlay_menu(self, popup_name) -> None:
        """Opens the TsdFrame overlay selection dialog."""
        dialog = TsdFramesDialog(self._tsdframes_model, parent=self)
        dialog.show()

    def show_select_iset_menu(self) -> None:
        """Opens the interval set selection dialog."""
        dialog = IntervalSetsDialog(self._interval_sets_model, parent=self)
        dialog.show()

    def _popup_menu(self) -> None:
        """Opens a dropdown dialog based on selected action."""
        action: QAction | None = self.sender()
        popup_name = action.objectName()
        if popup_name == "select_interval_set":
            self.show_select_iset_menu()
            return
        if popup_name in ["overlay_time_series", "overlay_skeleton"]:
            self.show_overlay_menu(popup_name)
            return

        kwargs = get_popup_kwargs(popup_name, self, action)
        if kwargs is not None:
            dialog = DropdownDialog(**kwargs)
            dialog.setEnabled(True)
            dialog.show()

    def jump_next(self) -> None:
        """Jump to the next timestamp or start"""
        self.plot.jump_next()

    def jump_previous(self) -> None:
        """ Jump to the previous timestamp or start"""
        self.plot.jump_previous()

    def _request_draw(self) -> None:
        """Request a redraw of the plot when channel states change."""
        widget = self.sender()
        materials = get_plot_attribute(self.plot, "material")
        for index, val in getattr(widget, "checks", {}).items():
            materials[index].opacity = val
        self.plot.canvas.request_draw(self.plot.animate)

    def _set_interval_set(self, name, colors, alpha, checked) -> None:
        """Add or remove IntervalSet from the plot based on selection."""
        if checked:
            # If already present, update the parameters
            if name in self.plot._epochs:
                self.plot.update_interval_set(name, colors=colors, alpha=alpha)
                self.plot.canvas.request_draw(self.plot.animate)
            else:
                self.plot.add_interval_sets(self._interval_sets[name], colors=colors, alpha=alpha, labels=name)
        else:
            self.plot.remove_interval_set(name)
            self.plot.canvas.request_draw(self.plot.animate)

    def _overlay_time_series(self, name, color, markersize, thickness, checked) -> None:
        """Add or remove TsdFrame overlay from the plot based on selection."""
        if checked:
            # If already present, update the parameters
            if name in self.plot.points:
                self.plot.points[name].set_color(color)
                self.plot.points[name].set_markersize(markersize)
                self.plot.points[name].set_thickness(thickness)
                self.plot.canvas.request_draw(self.plot.animate)
            else:
                self.plot.superpose_points(self._tsdframes[name], color, markersize, thickness, label=name)
        else:
            if name in self.plot.points:
                if hasattr(self.plot.points[name], "lines"):
                    self.plot.scene.remove(self.plot.points[name].lines)
                self.plot.scene.remove(self.plot.points[name].points)
                del self.plot.points[name]
            self.plot.canvas.request_draw(self.plot.animate)



