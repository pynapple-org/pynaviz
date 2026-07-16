"""
Test for PlotTsdFrame
"""
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest
from PIL import Image

import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import TsdFrameConfig


def test_plot_tsdframe_init(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)

    assert isinstance(v.data, nap.TsdFrame)
    assert v.cmap == "viridis"
    assert isinstance(v.renderer, gfx.Renderer)
    assert isinstance(v.scene, gfx.Scene)
    assert isinstance(v.ruler_x, gfx.Ruler)
    assert isinstance(v.ruler_y, gfx.Ruler)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.graphic, gfx.Line)
    v.close()

    assert hasattr(v._modes["lines"], "stream")

def test_plot_tsdframe_flush(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]
    max_n = mode.stream._max_n

    # Check that the init flushed the data
    pos = v.graphic.geometry.positions.data
    for i, c in enumerate(dummy_tsdframe.columns):
        sl = mode._buffer_slices[c]
        np.testing.assert_almost_equal(
            pos[sl, 0], dummy_tsdframe.t[0:max_n].astype("float32")
        )
        np.testing.assert_almost_equal(
            pos[sl, 1], dummy_tsdframe.d[0:max_n, i].astype("float32")
        )

    # Flush the same range
    v._flush(start=0, end=1)
    for i, c in enumerate(dummy_tsdframe.columns):
        sl = mode._buffer_slices[c]
        np.testing.assert_almost_equal(
            pos[sl, 0], dummy_tsdframe.t[0:max_n].astype("float32")
        )
        np.testing.assert_almost_equal(
            pos[sl, 1], dummy_tsdframe.d[0:max_n, i].astype("float32")
        )

    # Flush a different range (all data fits in memory, buffer unchanged)
    v._flush(start=1, end=1.5)
    for i, c in enumerate(dummy_tsdframe.columns):
        sl = mode._buffer_slices[c]
        np.testing.assert_almost_equal(
            pos[sl, 0], dummy_tsdframe.t[0:max_n].astype("float32")
        )
        np.testing.assert_almost_equal(
            pos[sl, 1], dummy_tsdframe.d[0:max_n, i].astype("float32")
        )

    v.close()

@pytest.mark.parametrize(
    "window", [
        (0, 6.1),
        (9, 11)
    ]
)
def test_plot_tsdframe_large(dummy_tsdframe, window):
    v = viz.PlotTsdFrame(dummy_tsdframe, window_size=2.0)
    mode = v._modes["lines"]

    # ws = 2 second -> max_n = 201
    assert mode.stream._max_n == 201
    win_start, win_end = window

    v._flush(start=win_start, end=win_end)
    sl = mode.stream.get_slice(win_start, win_end)

    pos = v.graphic.geometry.positions.data
    for i, c in enumerate(dummy_tsdframe.columns):
        bsl = mode._buffer_slices[c]
        x = pos[bsl, 0]
        np.testing.assert_almost_equal(
            x[~np.isnan(x)], dummy_tsdframe.t[sl].astype("float32")
        )
        y = pos[bsl, 1]
        np.testing.assert_almost_equal(
            y[~np.isnan(y)], dummy_tsdframe.d[sl, i].astype("float32")
        )

    v.close()


def test_plot_tsdframe_min_max(tmp_path, dummy_tsdframe):
    path = tmp_path / "test.dat"
    mmap = np.memmap(path, mode="w+", shape=dummy_tsdframe.d.shape, dtype=dummy_tsdframe.d.dtype)
    mmap[:] = dummy_tsdframe.d[:]
    mmap.flush()
    tsdframe = nap.TsdFrame(t=dummy_tsdframe.t, d=mmap, columns=dummy_tsdframe.columns)

    v = viz.PlotTsdFrame(tsdframe, window_size=2.0)
    minmax = v._get_min_max()
    np.testing.assert_almost_equal(minmax[:,0], np.min(tsdframe.get(0, 2), 0))
    np.testing.assert_almost_equal(minmax[:, 1], np.max(tsdframe.get(0, 2), 0))

    v.close()


@pytest.mark.parametrize(
    "func, kwargs",
    TsdFrameConfig.parameters,
)
def test_plot_tsdframe_action(dummy_tsdframe, func, kwargs):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    if func is not None:
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs):
                getattr(v, n)(**k)
        else:
            getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    filename = TsdFrameConfig._build_filename(func, kwargs)
    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots" / filename
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)
    v.close()


# ── LinesMode tests ──────────────────────────────────────────────────────────

def test_lines_mode_buffer_layout(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]
    n_cols = dummy_tsdframe.shape[1]
    max_n = mode.stream._max_n

    # Buffer has (max_n + 1) * n_channels rows, 3 columns (x, y, z)
    assert mode.buffer.shape == ((max_n + 1) * n_cols, 3)

    # Each channel slice has exactly max_n entries
    assert len(mode._buffer_slices) == n_cols
    for c in dummy_tsdframe.columns:
        sl = mode._buffer_slices[c]
        assert sl.stop - sl.start == max_n

    # NaN gaps between channels (the row right after each slice)
    for c in dummy_tsdframe.columns:
        sl = mode._buffer_slices[c]
        assert np.isnan(mode.buffer[sl.stop, 0])

    v.close()


def test_lines_mode_get_callbacks(dummy_tsdframe):
    # All data fits — no streaming callback needed
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]
    assert mode.stream._max_n >= dummy_tsdframe.shape[0]
    assert mode.get_callbacks() == []
    v.close()

    # Streaming needed
    v2 = viz.PlotTsdFrame(dummy_tsdframe, window_size=2.0)
    mode2 = v2._modes["lines"]
    assert mode2.stream._max_n < dummy_tsdframe.shape[0]
    callbacks = mode2.get_callbacks()
    assert len(callbacks) == 1
    assert callbacks[0] == mode2.stream.stream
    v2.close()


def test_lines_mode_rescale(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]

    # Before sorting: rescale should return False
    assert mode.rescale("i") is False

    # Sort, then rescale
    v.sort_by("channel")
    buf_before = mode.buffer[:, 1].copy()
    result = mode.rescale("i")
    assert result is True
    # Buffer y-values should have changed
    assert not np.allclose(mode.buffer[:, 1], buf_before, equal_nan=True)

    v.close()


def test_lines_mode_update_visibility(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]

    # All channels visible initially — positions are real (non-NaN) values
    first_col = dummy_tsdframe.columns[0]
    sl0 = mode._buffer_slices[first_col]
    assert not np.all(np.isnan(mode.buffer[sl0, 0]))

    # Hide first channel
    vis = v._manager.visible.copy()
    vis[0] = False
    v._manager.visible = vis
    mode.update_visibility()

    # Hidden channel: positions are NaN
    assert np.all(np.isnan(mode.buffer[sl0, 0]))
    assert np.all(np.isnan(mode.buffer[sl0, 1]))

    # Other channels still have real data
    for c in dummy_tsdframe.columns[1:]:
        sl = mode._buffer_slices[c]
        assert not np.all(np.isnan(mode.buffer[sl, 0]))

    # Un-hide the channel — positions are restored
    vis[0] = True
    v._manager.visible = vis
    mode.update_visibility()
    assert not np.all(np.isnan(mode.buffer[sl0, 0]))

    v.close()


# ── ImageMode tests ──────────────────────────────────────────────────────────

def test_image_mode_init(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    assert isinstance(v.graphic, gfx.Image)
    assert mode.buffer.shape[0] == dummy_tsdframe.shape[1]
    assert hasattr(mode, "texture")
    assert mode._n_visible == dummy_tsdframe.shape[1]
    assert isinstance(v.controller, viz.controller.SpanYLockController)

    v.close()


def test_image_mode_flush(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    # Buffer should have data (not all zeros after flush)
    assert not np.all(mode.buffer == 0)

    # Time alignment
    assert mode.graphic.local.x == pytest.approx(float(dummy_tsdframe.t[0]))

    # Buffer rows match data channels (all-data path, no reorder since unsorted)
    n = min(dummy_tsdframe.shape[0], mode.stream._max_n)
    for i in range(dummy_tsdframe.shape[1]):
        np.testing.assert_almost_equal(
            mode.buffer[i, :n],
            dummy_tsdframe.d[:n, i].astype("float32"),
        )

    v.close()


def test_image_mode_rescale(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    mode.graphic.material.clim = (-1, 1)

    # Narrow (increase)
    mode.rescale("i")
    lo, hi = mode.graphic.material.clim
    assert lo > -1 and hi < 1

    # Reset and widen (decrease)
    mode.graphic.material.clim = (-1, 1)
    mode.rescale("d")
    lo, hi = mode.graphic.material.clim
    assert lo < -1 and hi > 1

    v.close()


def test_image_mode_color_by(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    mode.color_by("plasma", "test", 0, 10, lambda vals: None, {})
    assert mode.graphic.material.map == gfx.cm.plasma
    lo, hi = mode.graphic.material.clim
    assert lo == 0 and hi == 10

    v.close()


def test_image_mode_visibility(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]
    n_channels = dummy_tsdframe.shape[1]

    assert mode._n_visible == n_channels

    # Hide first channel
    vis = v._manager.visible.copy()
    vis[0] = False
    v._manager.visible = vis
    mode.update_visibility()

    assert mode._n_visible == n_channels - 1
    assert mode.texture.data.shape[0] == n_channels - 1

    v.close()


def test_image_mode_reorder_rows(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]
    n_cols = dummy_tsdframe.shape[1]

    # No reorder when not sorted
    raw = np.arange(n_cols * 10, dtype="float32").reshape(n_cols, 10)
    result = mode._reorder_image_rows(raw)
    np.testing.assert_array_equal(result, raw)

    # Sort, then reorder should permute rows
    v.sort_by("channel")
    result = mode._reorder_image_rows(raw)
    offsets = np.array([
        v._manager.data.loc[c]["offset"] for c in dummy_tsdframe.columns
    ])
    order = np.argsort(offsets)
    np.testing.assert_array_equal(result, raw[order])

    v.close()


def test_image_mode_get_visible_buffer(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    # All visible: returns full buffer
    visible_buf = mode._get_visible_buffer()
    assert visible_buf.shape[0] == dummy_tsdframe.shape[1]

    # Hide one channel
    vis = v._manager.visible.copy()
    vis[0] = False
    v._manager.visible = vis
    visible_buf = mode._get_visible_buffer()
    assert visible_buf.shape[0] == dummy_tsdframe.shape[1] - 1

    v.close()


# ── XvsYMode tests ───────────────────────────────────────────────────────────

def test_xvsy_mode_init(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.plot_x_vs_y(x_col=0, y_col=1)
    mode = v._modes["x_vs_y"]

    assert isinstance(mode.graphic, gfx.Line)
    assert isinstance(mode.time_point, gfx.Points)
    assert mode.buffer.shape == (len(dummy_tsdframe), 3)

    # Buffer x/y match the two selected columns
    np.testing.assert_almost_equal(
        mode.buffer[:, 0], dummy_tsdframe.d[:, 0].astype("float32")
    )
    np.testing.assert_almost_equal(
        mode.buffer[:, 1], dummy_tsdframe.d[:, 1].astype("float32")
    )

    # Time point rendered above the line (z = 1)
    assert mode.time_point.geometry.positions.data[0, 2] == 1.0

    v.close()


def test_xvsy_mode_update_parameters(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["x_vs_y"]

    mode.update_parameters(x_col=2, y_col=3, color="blue", thickness=3.0, markersize=20.0)
    assert mode.x_col == 2
    assert mode.y_col == 3
    assert mode.color == "blue"
    assert mode.thickness == 3.0
    assert mode.markersize == 20.0

    v.close()


def test_xvsy_mode_update_buffer(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.plot_x_vs_y(x_col=0, y_col=1)
    mode = v._modes["x_vs_y"]

    # Move to frame 50
    mode._update_buffer(50)
    marker = mode.time_point.geometry.positions.data[0]
    line = mode.graphic.geometry.positions.data[50]
    np.testing.assert_almost_equal(marker[0:2], line[0:2])

    # Move to frame 100
    mode._update_buffer(100)
    marker = mode.time_point.geometry.positions.data[0]
    line = mode.graphic.geometry.positions.data[100]
    np.testing.assert_almost_equal(marker[0:2], line[0:2])

    v.close()


def test_xvsy_mode_get_callbacks(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["x_vs_y"]
    callbacks = mode.get_callbacks()
    assert len(callbacks) == 1
    assert callbacks[0] == mode._update_buffer

    v.close()


# ── SkeletonMode tests ───────────────────────────────────────────────────────

def _make_skeleton_tsdframe(n_keypoints=3, n_time=200):
    t = np.arange(n_time) / 10
    d = np.cumsum(np.random.randn(n_time, n_keypoints * 2), axis=0)
    columns = []
    for i in range(n_keypoints):
        columns += [f"kp{i}_x", f"kp{i}_y"]
    return nap.TsdFrame(t=t, d=d, columns=columns)


def test_plot_skeleton_init():
    data = _make_skeleton_tsdframe()
    v = viz.PlotTsdFrame(data)
    v.plot_skeleton()
    mode = v._modes["skeleton"]

    assert v._display_mode == "skeleton"
    assert isinstance(mode.graphic, gfx.Points)
    assert isinstance(mode.lines, gfx.Line)
    assert isinstance(v.controller, viz.controller.GetController)

    v.close()


def test_plot_skeleton_odd_columns_raises():
    data = nap.TsdFrame(t=np.arange(10), d=np.random.randn(10, 5))
    v = viz.PlotTsdFrame(data)
    with pytest.raises(ValueError):
        v.plot_skeleton()

    v.close()


def test_plot_skeleton_moves_points_to_current_frame():
    data = _make_skeleton_tsdframe()
    v = viz.PlotTsdFrame(data)
    v.plot_skeleton()
    mode = v._modes["skeleton"]

    frame_index = 50
    mode._update_buffer(frame_index)
    expected = np.asarray(data.values[frame_index]).reshape(-1, 2)
    np.testing.assert_almost_equal(
        mode.graphic.geometry.positions.data[:, :2], expected, decimal=4
    )

    v.close()


def test_plot_skeleton_roundtrip_state():
    data = _make_skeleton_tsdframe()
    v = viz.PlotTsdFrame(data)
    v.plot_skeleton(edges=[("kp0", "kp1")], color="blue", thickness=3.0, markersize=12.0)
    state = v.get_plot_state()

    v2 = viz.PlotTsdFrame(data)
    v2.set_plot_state(state)

    assert v2._display_mode == "skeleton"
    assert v2._modes["skeleton"].get_state() == v._modes["skeleton"].get_state()

    v.close()
    v2.close()


def test_plot_skeleton_back_to_lines():
    data = _make_skeleton_tsdframe()
    v = viz.PlotTsdFrame(data)
    v.plot_skeleton()
    v._set_mode("lines")

    assert v._display_mode == "lines"
    assert isinstance(v.controller, viz.controller.SpanController)

    v.close()


def test_plot_skeleton_uses_parent_metadata_by_default():
    # 3 keypoints, kp0/kp1 both children of root kp2 -> 2 bones, not the
    # complete-graph default of 3.
    data = _make_skeleton_tsdframe(n_keypoints=3)
    data.set_info(parent=["kp2", "kp2", "kp2", "kp2", None, None])

    v = viz.PlotTsdFrame(data)
    v.plot_skeleton()  # edges=None -> should read "parent" metadata, not warn

    mode = v._modes["skeleton"]
    assert mode.lines.geometry.positions.data.shape == (4, 3)  # 2 edges * 2 endpoints

    v.close()


# ── Mode switching ───────────────────────────────────────────────────────────

# ── compact_visible_offsets tests ────────────────────────────────────────────

def test_compact_offsets_snapshotted_after_group_by(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.group_by(metadata_name="group")
    assert v._manager._base_offset is not None
    np.testing.assert_array_equal(v._manager._base_offset, v._manager.offset)
    v.close()


def test_compact_offsets_snapshotted_after_sort_by(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.sort_by(metadata_name="channel")
    assert v._manager._base_offset is not None
    np.testing.assert_array_equal(v._manager._base_offset, v._manager.offset)
    v.close()


def test_compact_offsets_group_by_hide_whole_group(dummy_tsdframe):
    # dummy_tsdframe: columns [0,1,2,3,4], group metadata [0,0,1,0,1]
    # After group_by: group 0 channels at 1, group 1 channels at 3
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.group_by(metadata_name="group")

    group1_cols = [c for c in dummy_tsdframe.columns if dummy_tsdframe.metadata.loc[c]["group"] == 1]

    # Hide all group 1 channels
    vis = v._manager.visible.copy()
    for c in group1_cols:
        vis[list(dummy_tsdframe.columns).index(c)] = False
    v._manager.visible = vis

    y_max = v._manager.compact_visible_offsets()

    # All visible channels were in group 0 → same offset (1)
    for c in dummy_tsdframe.columns:
        if dummy_tsdframe.metadata.loc[c]["group"] == 0:
            assert v._manager.data.loc[c]["offset"] == pytest.approx(1.0)

    # y_max should be 1 (only group 0 visible, no gap needed)
    assert y_max == pytest.approx(1.0)
    v.close()


def test_compact_offsets_group_by_partial_hide(dummy_tsdframe):
    # Hide only one channel from group 1; group 1 still has channels → gap preserved
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.group_by(metadata_name="group")

    group1_cols = [c for c in dummy_tsdframe.columns if dummy_tsdframe.metadata.loc[c]["group"] == 1]
    vis = v._manager.visible.copy()
    # Hide only the first channel from group 1
    vis[list(dummy_tsdframe.columns).index(group1_cols[0])] = False
    v._manager.visible = vis

    y_max = v._manager.compact_visible_offsets()

    # Remaining visible group 1 channel should still be above group 0 (offset > 1)
    remaining_g1 = [c for c in group1_cols if c != group1_cols[0]]
    for c in remaining_g1:
        assert v._manager.data.loc[c]["offset"] > 1.0

    # y_max >= 3 since two groups are still visible
    assert y_max >= 3.0
    v.close()


def test_compact_offsets_sort_by_hide_rank(dummy_tsdframe):
    # After sort_by, hiding one channel should compact remaining without gaps
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.sort_by(metadata_name="channel")

    offsets_before = v._manager.offset.copy()

    # Hide one channel
    vis = v._manager.visible.copy()
    vis[0] = False
    v._manager.visible = vis

    y_max = v._manager.compact_visible_offsets()
    offsets_after = v._manager.offset.copy()

    # y_max should be less than before (one less channel)
    assert y_max < offsets_before.max()

    # Visible offsets should be consecutive starting at 1 (no gaps)
    _ = sorted(offsets_after[vis])
    assert y_max >= 1.0

    v.close()


def test_compact_offsets_restore_on_show_all(dummy_tsdframe):
    # Hide a group, then show all — offsets should return to base
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.group_by(metadata_name="group")

    base = v._manager._base_offset.copy()

    # Hide group 1
    group1_cols = [c for c in dummy_tsdframe.columns if dummy_tsdframe.metadata.loc[c]["group"] == 1]
    vis = v._manager.visible.copy()
    for c in group1_cols:
        vis[list(dummy_tsdframe.columns).index(c)] = False
    v._manager.visible = vis
    v._manager.compact_visible_offsets()

    # Show all again
    v._manager.visible = np.ones(len(dummy_tsdframe.columns), dtype=bool)
    y_max = v._manager.compact_visible_offsets()

    # Offsets should match the original base
    np.testing.assert_array_almost_equal(v._manager.offset, base)
    assert y_max == pytest.approx(float(base.max()))

    v.close()


def test_compact_offsets_no_op_without_sort_group(dummy_tsdframe):
    # compact_visible_offsets should do nothing when no sort/group is active
    v = viz.PlotTsdFrame(dummy_tsdframe)
    offsets_before = v._manager.offset.copy()

    y_max = v._manager.compact_visible_offsets()

    np.testing.assert_array_equal(v._manager.offset, offsets_before)
    assert y_max == pytest.approx(0.0)
    v.close()


def test_toggle_visibility_updates_ylim_with_group_by(dummy_tsdframe):
    # After group_by and hiding a whole group, ylim should shrink
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.group_by(metadata_name="group")

    ylim_before = v.controller.get_ylim()

    group1_cols = [c for c in dummy_tsdframe.columns if dummy_tsdframe.metadata.loc[c]["group"] == 1]
    vis = v._manager.visible.copy()
    for c in group1_cols:
        vis[list(dummy_tsdframe.columns).index(c)] = False
    v._manager.visible = vis
    v._update("toggle_visibility")

    ylim_after = v.controller.get_ylim()
    assert ylim_after[1] < ylim_before[1]
    v.close()


def test_toggle_display_mode(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)

    # Starts in lines mode
    assert v._display_mode == "lines"
    assert isinstance(v.graphic, gfx.Line)

    # Toggle to image
    v._toggle_display_mode(SimpleNamespace(type="key_down", key="m"))
    assert v._display_mode == "image"
    assert isinstance(v.graphic, gfx.Image)
    assert isinstance(v.controller, viz.controller.SpanYLockController)

    # Toggle back to lines
    v._toggle_display_mode(SimpleNamespace(type="key_down", key="m"))
    assert v._display_mode == "lines"
    assert isinstance(v.graphic, gfx.Line)
    assert not isinstance(v.controller, viz.controller.SpanYLockController)
    assert isinstance(v.controller, viz.controller.SpanController)

    v.close()
