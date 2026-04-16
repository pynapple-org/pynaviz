"""
Dock widget containing the list of variables. Double-clicking a variable adds it to the main view.
The tree structure of the variables is preserved in the dock, allowing for easy navigation and organization of complex datasets.
"""

import pynapple as nap
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QCursor, QFontMetrics
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

DOCK_LIST_STYLESHEET = """
    * {
        border : 2px solid black;
        background : #272822;
        color : #F8F8F2;
        selection-color : yellow;
        selection-background-color : #E69F66;
    }
"""


def get_children_dict(parent: QTreeWidget | QTreeWidgetItem):
    """Helper function to get children as a dictionary."""
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


def _get_item_key_path(item: QTreeWidgetItem, key_path: None | list = None) -> list:
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
            "Reset view: r\n"
            "Pan left/right by one page: <- / ->\n"
            "Jump to next/prev superposed epoch: Ctrl+-> / Ctrl+<-\n"
            "Lock/unlock y-axis: y\n"
            "Lock/unlock x-axis: x\n\n"
            "Specific to TsdFrame:\n"
            "Increase contrast: i\n"
            "Decrease contrast: d\n\n"
            "Specific to TsGroup:\n"
            "Increase marker size: i\n"
            "Decrease marker size: d\n\n"
            "Specific to IntervalSet & Timestamps:\n"
            "Jump to next interval/timestamp: n or Ctrl+->\n"
            "Jump to previous interval/timestamp: p or Ctrl+<-\n"
        )

        # Frameless floating box
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)

        # Layout with help text
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(label)

        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button.setFixedWidth(100)
        close_button.setStyleSheet("margin-top: 10px;")
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

        # Track clicks outside
        if parent:
            parent.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            # Close the help box if clicking outside
            if not self.geometry().contains(event.globalPosition().toPoint()):
                self.close()
        return super().eventFilter(obj, event)


class VariableDock(QDockWidget):
    def __init__(self, variables, gui):
        """
        Sidebar widget containing the list of variables.

        Parameters
        ----------
        variables : dict
            Dictionary of pynapple variables.
        gui : QMainWindow
            Reference to the main GUI instance.
        """
        super().__init__("Variables", gui)
        self.gui = gui
        self.variables = variables
        self._interval_set_key_paths = []
        self.setObjectName("VariablesDock")
        self.setStyleSheet(DOCK_LIST_STYLESHEET)

        self.collapsed_width = 20
        self.expanded = True

        # --- Dock settings ---
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setFeatures(
            # QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        # --- Main container inside dock ---
        container = QWidget()
        main_layout = QHBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Content area ---
        self.content = QWidget()
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Tree widget
        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self._add_items_to_tree_widget(variables)
        content_layout.addWidget(self.treeWidget)

        main_layout.addWidget(self.content)

        # --- Handle for collapsing ---
        self.handle = QPushButton("◀")
        self.handle.setFixedWidth(20)
        self.handle.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.handle.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: #d0d0d0;
            }
            QPushButton:hover {
                background-color: #c0c0c0;
            }
        """)
        self.handle.clicked.connect(self.toggle)
        main_layout.addWidget(self.handle)

        # Set container as dock widget content
        self.setWidget(container)

        self._resize_dock()

    def _resize_dock(self):
        """Resize the dock to fit the widest visible tree item, including nested ones."""
        self.treeWidget.expandAll()
        indent = self.treeWidget.indentation()
        metrics = QFontMetrics(self.treeWidget.font())

        def _item_width(item, depth):
            w = depth * indent + metrics.horizontalAdvance(item.text(0)) + indent
            for i in range(item.childCount()):
                w = max(w, _item_width(item.child(i), depth + 1))
            return w

        max_w = 100
        for i in range(self.treeWidget.topLevelItemCount()):
            max_w = max(max_w, _item_width(self.treeWidget.topLevelItem(i), 0))

        self.expanded_width = max_w + 50  # 20px for the collapse handle
        self.last_width = self.expanded_width
        self.setFixedWidth(self.expanded_width)

    def toggle(self):
        """Collapse or expand the dock content."""
        if self.expanded:
            self.content.hide()
            self.last_width = self.width()
            self.setFixedWidth(self.collapsed_width)
            self.handle.setText("▶")
        else:
            self.content.show()
            self.setFixedWidth(self.expanded_width)
            self.resize(self.last_width, self.height())
            self.handle.setText("◀")
        self.expanded = not self.expanded

    def _add_items_to_tree_widget(self, item_dict: dict, parent: None | QTreeWidgetItem = None, clear: bool = False):
        """Recursively add items to the tree widget from a nested dictionary.

        Parameters
        ----------
        item_dict:
            Nested dictionary representing the tree structure.
        parent:
            Parent QTreeWidgetItem to which items will be added. If None, items are added to the top level.
        clear:
            If True, clear the existing items in the tree widget before adding new ones.
        """
        if clear:
            self.treeWidget.clear()
            self._interval_set_key_paths = []

        if parent is None:
            parent = self.treeWidget

        # Get existing children
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

        self._resize_dock()

    def on_item_double_clicked(self, item, column):
        """Handle double-click only for leaf items."""
        if item.childCount() == 0:  # This is a leaf item
            self.add_dock_widget(item)

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

    def add_dock_widget(self, item: QTreeWidgetItem | list[str]) -> QDockWidget | None:
        key_path = self._extract_variable_path(item)
        if not key_path:
            return None

        variable = self._get_variable(key_path)
        if variable is None:
            return

        self.gui.add_dock_widget(variable, key_path)
