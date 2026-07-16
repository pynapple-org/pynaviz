import numpy as np
import pynapple as nap
import pytest

from pynaviz.display_modes import SkeletonMode


def _make_tsdframe(n_keypoints, n_time=5, named=True):
    n_cols = n_keypoints * 2
    d = np.arange(n_time * n_cols, dtype=float).reshape(n_time, n_cols)
    if named:
        columns = []
        for i in range(n_keypoints):
            columns += [f"kp{i}_x", f"kp{i}_y"]
    else:
        columns = None
    kwargs = {"t": np.arange(n_time), "d": d}
    if columns is not None:
        kwargs["columns"] = columns
    return nap.TsdFrame(**kwargs)


def test_skeleton_mode_labels_from_named_columns():
    data = _make_tsdframe(3)
    mode = SkeletonMode(data, manager=None)
    assert mode.labels == ["kp0", "kp1", "kp2"]


def test_skeleton_mode_initialize_graphic_two_points_no_warning():
    data = _make_tsdframe(2)
    mode = SkeletonMode(data, manager=None)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mode.initialize_graphic()

    assert mode.graphic is not None
    assert mode.lines is not None
    assert mode.graphic.geometry.positions.data.shape == (2, 3)
    # 1 edge * 2 endpoints
    assert mode.lines.geometry.positions.data.shape == (2, 3)


def test_skeleton_mode_initialize_graphic_complete_graph_warns():
    data = _make_tsdframe(3)
    mode = SkeletonMode(data, manager=None)
    with pytest.warns(UserWarning):
        mode.initialize_graphic()
    # complete graph over 3 points -> 3 edges -> 6 rows
    assert mode.lines.geometry.positions.data.shape == (6, 3)


def test_skeleton_mode_uses_parent_metadata_when_edges_none():
    data = _make_tsdframe(3)
    # kp0, kp1 -> child of kp2 (root); star topology, 2 edges not 3
    data.set_info(parent=["kp2", "kp2", "kp2", "kp2", None, None])
    mode = SkeletonMode(data, manager=None)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mode.initialize_graphic()  # must not warn: metadata edges found, no complete-graph fallback

    # 2 edges (kp0-kp2, kp1-kp2) * 2 endpoints = 4 rows
    assert mode.lines.geometry.positions.data.shape == (4, 3)


def test_skeleton_mode_explicit_edges_override_parent_metadata():
    data = _make_tsdframe(3)
    data.set_info(parent=["kp2", "kp2", "kp2", "kp2", None, None])
    mode = SkeletonMode(data, manager=None)
    mode.update_parameters(edges=[("kp0", "kp1")])
    mode.initialize_graphic()

    assert mode.lines.geometry.positions.data.shape == (2, 3)


def test_skeleton_mode_explicit_edges_by_name():
    data = _make_tsdframe(3)
    mode = SkeletonMode(data, manager=None)
    mode.update_parameters(edges=[("kp0", "kp1")])
    mode.initialize_graphic()
    assert mode.lines.geometry.positions.data.shape == (2, 3)


def test_skeleton_mode_update_buffer_moves_points():
    data = _make_tsdframe(2)
    mode = SkeletonMode(data, manager=None)
    mode.initialize_graphic()

    frame_index = 3
    mode._update_buffer(frame_index)

    expected = np.asarray(data.values[frame_index]).reshape(-1, 2)
    np.testing.assert_array_equal(
        mode.graphic.geometry.positions.data[:, :2], expected
    )


def test_skeleton_mode_update_buffer_calls_request_draw():
    data = _make_tsdframe(2)
    mode = SkeletonMode(data, manager=None)
    mode.initialize_graphic()

    calls = []
    mode._request_draw = lambda: calls.append(True)
    mode._update_buffer(0)
    assert calls == [True]


def test_skeleton_mode_get_callbacks_returns_update_buffer():
    data = _make_tsdframe(2)
    mode = SkeletonMode(data, manager=None)
    callbacks = mode.get_callbacks()
    assert callbacks == [mode._update_buffer]


def test_skeleton_mode_rescale_is_noop():
    data = _make_tsdframe(2)
    mode = SkeletonMode(data, manager=None)
    mode.initialize_graphic()
    assert mode.rescale("i") is False


def test_skeleton_mode_state_roundtrip():
    data = _make_tsdframe(2)
    mode = SkeletonMode(data, manager=None, default_color="blue")
    mode.update_parameters(edges=[("kp0", "kp1")], color="green", thickness=3.0, markersize=12.0)

    state = mode.get_state()
    assert state == {
        "window_size": None,
        "edges": [("kp0", "kp1")],
        "color": "green",
        "thickness": 3.0,
        "markersize": 12.0,
    }

    mode2 = SkeletonMode(data, manager=None)
    mode2.set_state(state)
    assert mode2.get_state() == state


def test_skeleton_mode_positional_labels_fallback():
    data = _make_tsdframe(2, named=False)
    mode = SkeletonMode(data, manager=None)
    assert mode.labels == ["0", "1"]


def test_skeleton_mode_construction_does_not_validate_columns():
    # PlotTsdFrame builds a SkeletonMode for every TsdFrame regardless of
    # column count; label resolution (and its even-column requirement) must
    # be deferred until skeleton mode is actually used.
    data = nap.TsdFrame(t=np.arange(5), d=np.random.randn(5, 3))
    mode = SkeletonMode(data, manager=None)  # must not raise
    with pytest.raises(ValueError):
        _ = mode.labels


def test_skeleton_mode_uses_dedicated_controller_key():
    # Must differ from XvsYMode's "get" so the two modes don't fight over
    # the same GetController's plot_callbacks.
    assert SkeletonMode.controller_key == "skeleton_get"


def test_skeleton_mode_get_extent():
    data = _make_tsdframe(2, n_time=4)
    mode = SkeletonMode(data, manager=None)
    xmin, xmax, ymin, ymax = mode.get_extent()

    values = np.asarray(data.values)
    assert xmin == np.nanmin(values[:, 0::2])
    assert xmax == np.nanmax(values[:, 0::2])
    assert ymin == np.nanmin(values[:, 1::2])
    assert ymax == np.nanmax(values[:, 1::2])
