"""
Utils functions for creating demos and tutorials.
"""
import sys
from pathlib import Path

import imageio
import numpy as np
import pynapple as nap
import requests, os
from PyQt6.QtGui import QAction
from one.api import ONE

import pynaviz as viz
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDockWidget, QMenu, QDialog
from pynaviz.qt.mainwindow import MainWindow
from matplotlib.pyplot import *
from PIL import ImageGrab, Image


def get_action_by_object_name(menu: QMenu, name: str) -> QAction | None:
    for action in menu.actions():
        if action.objectName() == name:
            return action
    return None

def grab_window(win):
    """Capture the visible pixels of the Qt window and return a PIL image."""
    rect = win.frameGeometry()
    top_left = rect.topLeft()
    x, y, w, h = top_left.x(), top_left.y(), rect.width(), rect.height()
    bbox = (x, y, x + w, y + h)
    return ImageGrab.grab(bbox=bbox)

def click_on_item(list_widget, item_number, win, app, frames, durations):
    item = list_widget.item(item_number)
    rect = list_widget.visualItemRect(item)
    pos = rect.center()
    QTest.mouseClick(list_widget.viewport(),
                     Qt.MouseButton.LeftButton,
                     pos=pos)
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    return item

def click_on_tree_item(tree_widget, item_index, win, app, frames, durations, column=0):
    """Click on a tree widget item by index.

    Parameters
    ----------
    tree_widget : QTreeWidget
        The tree widget
    item_index : int or tuple
        If int, gets top-level item at that index.
        If tuple like (0, 1), gets child item: topLevelItem(0).child(1)
    column : int
        Column to click on
    """
    # Get the item
    if isinstance(item_index, int):
        item = tree_widget.topLevelItem(item_index)
    elif isinstance(item_index, tuple):
        item = tree_widget.topLevelItem(item_index[0])
        for child_idx in item_index[1:]:
            item = item.child(child_idx)
    else:
        raise ValueError("item_index must be int or tuple")

    if item is None:
        raise ValueError(f"Item not found at index {item_index}")

    # Get visual rectangle and click
    rect = tree_widget.visualItemRect(item)
    pos = rect.center()
    QTest.mouseClick(tree_widget.viewport(),
                     Qt.MouseButton.LeftButton,
                     pos=pos)
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    return item

def add_dock_widget(tree_widget, win, app, frames, durations, item_number=0):
    item = click_on_tree_item(tree_widget, item_number, win, app, frames, durations)
    app.processEvents()
    tree_widget.itemDoubleClicked.emit(item, 0)
    app.processEvents()
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)

def flash_widget(widget, duration=200):
    orig_style = widget.styleSheet()
    widget.setStyleSheet("border: 2px solid red;")  # highlight
    QTimer.singleShot(duration, lambda: widget.setStyleSheet(orig_style))

def click_on_action(action_text, win, frames, durations):
    frames.append(grab_window(win))
    durations.append(800)
    menu = next((w for w in QApplication.topLevelWidgets()
                 if isinstance(w, QMenu) and w.isVisible()), None)
    # Find the QAction by visible text
    act = next((a for a in menu.actions() if a.text().replace("&", "") == action_text), None)
    if not act:
        raise ValueError(f"Menu item '{action_text}' not found")
    # Click inside the action rectangle
    rect = menu.actionGeometry(act)
    # Schedule dialog handling after a short delay
    QTimer.singleShot(1000, lambda: dialog_selection(win, frames, durations))
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=rect.center())
    # Wait for dialog to appear and handle it

def dialog_selection(win, frames, durations):
    frames.append(grab_window(win))
    durations.append(800)

    dialog = next((w for w in QApplication.topLevelWidgets()
                   if isinstance(w, QDialog) and w.isVisible()), None)
    if dialog is None:
        raise RuntimeError("No dialog found")
    # Find the OK button
    ok_button = dialog.ok_button
    ok_button.setStyleSheet("border: 2px solid red;")
    frames.append(grab_window(win))
    durations.append(800)
    QTest.mouseClick(ok_button, Qt.MouseButton.LeftButton)

def move_and_resize_dock(win, app, frames, durations, move_offset=QPoint(500, 0), resize_width=400):
    """
    Moves the last added dock widget by a given offset and resizes it.
    Captures frames before and after the move/resize.
    """
    dock_widgets = win.findChildren(QDockWidget)
    dock = dock_widgets[-1]  # last added dock (video)
    title_bar = dock.titleBarWidget()
    start = title_bar.mapToGlobal(title_bar.rect().center())
    end = start + move_offset
    QTest.mousePress(title_bar, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier,
                     pos=title_bar.rect().center())

    # Process events so the dock enters drag mode
    QApplication.processEvents()

    # Move to target
    QTest.mouseMove(title_bar, pos=title_bar.mapFromGlobal(end))
    QApplication.processEvents()

    # Release to drop
    QTest.mouseRelease(title_bar, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier,
                       pos=title_bar.mapFromGlobal(end))
    QTest.qWait(1000)
    app.processEvents()

    # Resize the dock
    win.resizeDocks([dock], [resize_width], Qt.Orientation.Horizontal)
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    app.processEvents()

def save_gif(frames, durations, filename="output.gif"):
    w, h = frames[0].size
    new_size = (int(w * 0.75), int(h * 0.75))
    frames = [im.resize(new_size, Image.LANCZOS)
                      for im in frames]
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=durations,
        loop=0
    )