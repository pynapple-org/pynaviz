"""Tests for GroupedChannelTree and GroupedChannelList, and MenuWidget.show_select_menu
branching when group_by is active.
"""

import numpy as np
import pynapple as nap
import pytest
from PySide6.QtCore import Qt

import pynaviz.qt.mainwindow as mw
from pynaviz.qt.widget_list_selection import (
    ChannelList,
    GroupedChannelList,
    GroupedChannelTree,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tsdframe(n_channels=4):
    t = np.arange(0, 1, 0.01)
    metadata = {"group": [0, 0, 1, 1][:n_channels], "rank": list(range(n_channels))}
    return nap.TsdFrame(
        t=t,
        d=np.random.randn(len(t), n_channels),
        columns=[f"ch{i}" for i in range(n_channels)],
        metadata=metadata,
    )


def _tree_fixture(groups_mapping=None, order_mapping=None, visibility_mapping=None):
    channels = ["ch0", "ch1", "ch2", "ch3"]
    if groups_mapping is None:
        groups_mapping = {"ch0": "A", "ch1": "A", "ch2": "B", "ch3": "B"}
    if visibility_mapping is None:
        visibility_mapping = {ch: True for ch in channels}
    return channels, groups_mapping, order_mapping, visibility_mapping


# ---------------------------------------------------------------------------
# GroupedChannelTree — structure
# ---------------------------------------------------------------------------

class TestGroupedChannelTreeStructure:

    def test_top_level_count_matches_unique_groups(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)
        assert tree.topLevelItemCount() == 2

    def test_group_labels_are_correct(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)
        labels = {tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())}
        assert labels == {"A", "B"}

    def test_child_count_per_group(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)
        counts = {
            tree.topLevelItem(i).text(0): tree.topLevelItem(i).childCount()
            for i in range(tree.topLevelItemCount())
        }
        assert counts == {"A": 2, "B": 2}

    def test_leaf_names_match_channels(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)
        leaf_names = set()
        for i in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(i)
            for j in range(group.childCount()):
                leaf_names.add(group.child(j).text(0))
        assert leaf_names == set(channels)

    def test_channels_ordered_by_order_mapping(self, qtbot):
        channels = ["ch0", "ch1", "ch2", "ch3"]
        gmap = {"ch0": "A", "ch1": "A", "ch2": "A", "ch3": "A"}
        omap = {"ch0": 3, "ch1": 1, "ch2": 0, "ch3": 2}
        vmap = {ch: True for ch in channels}
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)
        group = tree.topLevelItem(0)
        leaf_names = [group.child(j).text(0) for j in range(group.childCount())]
        assert leaf_names == ["ch2", "ch1", "ch3", "ch0"]

    def test_initial_visibility_all_checked(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)
        vis = tree._visibility_array()
        assert vis.all()

    def test_initial_visibility_partial(self, qtbot):
        channels = ["ch0", "ch1", "ch2", "ch3"]
        gmap = {"ch0": "A", "ch1": "A", "ch2": "B", "ch3": "B"}
        vmap = {"ch0": True, "ch1": False, "ch2": True, "ch3": False}
        tree = GroupedChannelTree(channels, gmap, None, vmap)
        qtbot.addWidget(tree)
        vis = tree._visibility_array()
        np.testing.assert_array_equal(vis, [True, False, True, False])

    def test_group_with_all_hidden_shows_unchecked(self, qtbot):
        channels = ["ch0", "ch1", "ch2", "ch3"]
        gmap = {"ch0": "A", "ch1": "A", "ch2": "B", "ch3": "B"}
        vmap = {"ch0": False, "ch1": False, "ch2": True, "ch3": True}
        tree = GroupedChannelTree(channels, gmap, None, vmap)
        qtbot.addWidget(tree)
        # find group A
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            if item.text(0) == "A":
                assert item.checkState(0) == Qt.CheckState.Unchecked
            else:
                assert item.checkState(0) == Qt.CheckState.Checked

    def test_group_with_mixed_shows_partial(self, qtbot):
        channels = ["ch0", "ch1", "ch2", "ch3"]
        gmap = {"ch0": "A", "ch1": "A", "ch2": "B", "ch3": "B"}
        vmap = {"ch0": True, "ch1": False, "ch2": True, "ch3": True}
        tree = GroupedChannelTree(channels, gmap, None, vmap)
        qtbot.addWidget(tree)
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            if item.text(0) == "A":
                assert item.checkState(0) == Qt.CheckState.PartiallyChecked


# ---------------------------------------------------------------------------
# GroupedChannelTree — interaction
# ---------------------------------------------------------------------------

class TestGroupedChannelTreeInteraction:

    def _find_group(self, tree, label):
        for i in range(tree.topLevelItemCount()):
            if tree.topLevelItem(i).text(0) == label:
                return tree.topLevelItem(i)
        return None

    def test_uncheck_group_unchecks_all_children(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)

        group_a = self._find_group(tree, "A")
        group_a.setCheckState(0, Qt.CheckState.Unchecked)

        for j in range(group_a.childCount()):
            assert group_a.child(j).checkState(0) == Qt.CheckState.Unchecked

    def test_check_group_checks_all_children(self, qtbot):
        channels = ["ch0", "ch1", "ch2", "ch3"]
        gmap = {"ch0": "A", "ch1": "A", "ch2": "B", "ch3": "B"}
        vmap = {"ch0": False, "ch1": False, "ch2": True, "ch3": True}
        tree = GroupedChannelTree(channels, gmap, None, vmap)
        qtbot.addWidget(tree)

        group_a = self._find_group(tree, "A")
        group_a.setCheckState(0, Qt.CheckState.Checked)

        for j in range(group_a.childCount()):
            assert group_a.child(j).checkState(0) == Qt.CheckState.Checked

    def test_visibility_array_after_uncheck_group(self, qtbot):
        channels = ["ch0", "ch1", "ch2", "ch3"]
        gmap = {"ch0": "A", "ch1": "A", "ch2": "B", "ch3": "B"}
        vmap = {ch: True for ch in channels}
        tree = GroupedChannelTree(channels, gmap, None, vmap)
        qtbot.addWidget(tree)

        self._find_group(tree, "A").setCheckState(0, Qt.CheckState.Unchecked)

        vis = tree._visibility_array()
        # ch0, ch1 → False; ch2, ch3 → True
        np.testing.assert_array_equal(vis, [False, False, True, True])

    def test_leaf_toggle_updates_parent_to_partial(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)

        group_a = self._find_group(tree, "A")
        group_a.child(0).setCheckState(0, Qt.CheckState.Unchecked)

        assert group_a.checkState(0) == Qt.CheckState.PartiallyChecked

    def test_leaf_toggle_all_unchecked_makes_parent_unchecked(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)

        group_a = self._find_group(tree, "A")
        for j in range(group_a.childCount()):
            group_a.child(j).setCheckState(0, Qt.CheckState.Unchecked)

        assert group_a.checkState(0) == Qt.CheckState.Unchecked

    def test_visibility_signal_emitted_on_leaf_change(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)

        emitted = []
        tree.visibilityChanged.connect(lambda v: emitted.append(v.copy()))

        group_a = self._find_group(tree, "A")
        group_a.child(0).setCheckState(0, Qt.CheckState.Unchecked)

        assert len(emitted) >= 1
        assert isinstance(emitted[-1], np.ndarray)
        assert emitted[-1].dtype == bool

    def test_visibility_signal_emitted_on_group_change(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        tree = GroupedChannelTree(channels, gmap, omap, vmap)
        qtbot.addWidget(tree)

        emitted = []
        tree.visibilityChanged.connect(lambda v: emitted.append(v.copy()))

        self._find_group(tree, "B").setCheckState(0, Qt.CheckState.Unchecked)

        assert len(emitted) >= 1
        assert not emitted[-1][2]  # ch2 → False
        assert not emitted[-1][3]  # ch3 → False


# ---------------------------------------------------------------------------
# GroupedChannelList dialog
# ---------------------------------------------------------------------------

class TestGroupedChannelList:

    def test_dialog_creates_tree(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        dialog = GroupedChannelList(channels, gmap, omap, vmap)
        qtbot.addWidget(dialog)
        assert isinstance(dialog.tree, GroupedChannelTree)

    def test_dialog_re_emits_visibility_changed(self, qtbot):
        channels, gmap, omap, vmap = _tree_fixture()
        dialog = GroupedChannelList(channels, gmap, omap, vmap)
        qtbot.addWidget(dialog)

        received = []
        dialog.visibilityChanged.connect(lambda v: received.append(v))

        group = dialog.tree.topLevelItem(0)
        group.setCheckState(0, Qt.CheckState.Unchecked)

        assert len(received) >= 1


# ---------------------------------------------------------------------------
# MenuWidget.show_select_menu — flat vs tree branching
# ---------------------------------------------------------------------------

class TestMenuWidgetShowSelectMenu:

    @pytest.fixture
    def window_tsdframe(self, qtbot):
        tsdframe = _make_tsdframe()
        variables = {"tsdframe": tsdframe}
        window = mw.MainWindow(variables)
        qtbot.addWidget(window)
        dock = window.add_dock_widget(tsdframe, ["tsdframe"])
        widget = dock.widget()
        return window, widget, tsdframe

    def test_flat_list_when_no_group_by(self, window_tsdframe, qtbot):
        _, widget, _ = window_tsdframe
        menu = widget.button_container
        # No group_by active → should open a ChannelList (flat)
        menu.show_select_menu()
        dialogs = [w for w in menu.findChildren(ChannelList)]
        # The dialog was created and shown
        assert len(dialogs) == 1
        assert not any(isinstance(w, GroupedChannelList) for w in menu.findChildren(GroupedChannelList))

    def test_tree_dialog_when_group_by_active(self, window_tsdframe, qtbot):
        _, widget, tsdframe = window_tsdframe
        menu = widget.button_container
        widget.plot.group_by(metadata_name="group")

        menu.show_select_menu()

        grouped_dialogs = menu.findChildren(GroupedChannelList)
        assert len(grouped_dialogs) == 1

    def test_tree_has_correct_groups_after_group_by(self, window_tsdframe, qtbot):
        _, widget, tsdframe = window_tsdframe
        menu = widget.button_container
        widget.plot.group_by(metadata_name="group")
        menu.show_select_menu()

        dialog = menu.findChildren(GroupedChannelList)[0]
        top_labels = {
            dialog.tree.topLevelItem(i).text(0)
            for i in range(dialog.tree.topLevelItemCount())
        }
        assert top_labels == {"0", "1"}

    def test_visibility_update_propagates_to_manager(self, window_tsdframe, qtbot):
        _, widget, _ = window_tsdframe
        menu = widget.button_container
        widget.plot.group_by(metadata_name="group")
        menu.show_select_menu()

        dialog = menu.findChildren(GroupedChannelList)[0]
        # Uncheck group "0" → ch0, ch1 should become hidden
        for i in range(dialog.tree.topLevelItemCount()):
            if dialog.tree.topLevelItem(i).text(0) == "0":
                dialog.tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Unchecked)
                break

        manager = widget.plot._manager
        # channels in group "0" are ch0 and ch1 (indices 0 and 1)
        assert not manager.visible[0]
        assert not manager.visible[1]
        # channels in group "1" remain visible
        assert manager.visible[2]
        assert manager.visible[3]
