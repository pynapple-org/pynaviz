import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest

from pynaviz.skeleton_geometry import (
    SkeletonGeometry,
    edges_from_parent_metadata,
    resolve_edges,
    resolve_keypoint_labels,
)

# ---------------------------------------------------------------------------
# resolve_keypoint_labels
# ---------------------------------------------------------------------------

def test_resolve_keypoint_labels_named():
    columns = ["Nose_x", "Nose_y", "EarL_x", "EarL_y"]
    assert resolve_keypoint_labels(columns) == ["Nose", "EarL"]


def test_resolve_keypoint_labels_positional_fallback():
    columns = [0, 1, 2, 3]
    assert resolve_keypoint_labels(columns) == ["0", "1"]


def test_resolve_keypoint_labels_mismatched_prefix_falls_back():
    # "_x"/"_y" suffixes but different prefixes -> not a matching pair
    columns = ["Nose_x", "EarL_y"]
    assert resolve_keypoint_labels(columns) == ["0"]


def test_resolve_keypoint_labels_odd_length_raises():
    with pytest.raises(ValueError):
        resolve_keypoint_labels(["Nose_x", "Nose_y", "EarL_x"])


# ---------------------------------------------------------------------------
# resolve_edges
# ---------------------------------------------------------------------------

def test_resolve_edges_default_is_complete_graph_and_warns():
    labels = ["A", "B", "C"]
    with pytest.warns(UserWarning):
        edges = resolve_edges(labels, edges=None)
    assert edges == [(0, 1), (0, 2), (1, 2)]


def test_resolve_edges_two_points_no_warning():
    labels = ["A", "B"]
    with warnings_none():
        edges = resolve_edges(labels, edges=None)
    assert edges == [(0, 1)]


def test_resolve_edges_explicit_names():
    labels = ["Nose", "EarL", "EarR"]
    edges = resolve_edges(labels, edges=[("Nose", "EarL"), ("Nose", "EarR")])
    assert edges == [(0, 1), (0, 2)]


def test_resolve_edges_explicit_indices():
    labels = ["Nose", "EarL", "EarR"]
    edges = resolve_edges(labels, edges=[(0, 1)])
    assert edges == [(0, 1)]


def test_resolve_edges_unknown_label_raises():
    labels = ["Nose", "EarL"]
    with pytest.raises(ValueError):
        resolve_edges(labels, edges=[("Nose", "Tail")])


# ---------------------------------------------------------------------------
# edges_from_parent_metadata
# ---------------------------------------------------------------------------

def _make_tsdframe_with_parents(parents=None):
    d = np.random.randn(10, 6)
    columns = ["Nose_x", "Nose_y", "EarL_x", "EarL_y", "Back1_x", "Back1_y"]
    data = nap.TsdFrame(t=np.arange(10), d=d, columns=columns)
    if parents is not None:
        data.set_info(parent=parents)
    return data


def test_edges_from_parent_metadata_no_metadata_returns_none():
    data = _make_tsdframe_with_parents(parents=None)
    assert edges_from_parent_metadata(data) is None


def test_edges_from_parent_metadata_no_parent_field_returns_none():
    data = _make_tsdframe_with_parents(parents=None)
    data.set_info(unrelated_field=["a"] * data.shape[1])
    assert edges_from_parent_metadata(data) is None


def test_edges_from_parent_metadata_returns_child_parent_pairs():
    data = _make_tsdframe_with_parents(
        parents=["Back1", "Back1", "Back1", "Back1", None, None]
    )
    edges = edges_from_parent_metadata(data)
    assert edges == [("Nose", "Back1"), ("EarL", "Back1")]


def test_edges_from_parent_metadata_feeds_resolve_edges():
    data = _make_tsdframe_with_parents(
        parents=["Back1", "Back1", "Back1", "Back1", None, None]
    )
    labels = resolve_keypoint_labels(data.columns)
    edges = edges_from_parent_metadata(data)
    assert resolve_edges(labels, edges) == [(0, 2), (1, 2)]


class warnings_none:
    """Context manager asserting no warnings were raised."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings(record=True)
        self._records = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        assert len(self._records) == 0
        return self._cm.__exit__(*exc)


# ---------------------------------------------------------------------------
# SkeletonGeometry
# ---------------------------------------------------------------------------

@pytest.fixture
def xy3():
    return np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype="float32")


def test_skeleton_geometry_bones_use_segment_material_not_polyline():
    # Two disjoint bones sharing no endpoint. A plain gfx.Line (continuous
    # polyline) would draw a spurious extra segment connecting the end of the
    # first bone to the start of the second (the reported "loops" artifact).
    # LineSegmentMaterial draws each consecutive pair independently instead.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]], dtype="float32")
    geo = SkeletonGeometry(xy, edges=[(0, 1), (2, 3)])
    assert isinstance(geo.lines.material, gfx.LineSegmentMaterial)


def test_skeleton_geometry_construction(xy3):
    edges = [(0, 1), (1, 2)]
    geo = SkeletonGeometry(xy3, edges, color="red", markersize=8.0, thickness=2.0)

    assert geo.n_points == 3
    np.testing.assert_array_equal(
        geo.points.geometry.positions.data[:, :2], xy3
    )
    assert geo.lines is not None
    # 2 edges * 2 endpoints each = 4 rows
    assert geo.lines.geometry.positions.data.shape == (4, 3)


def test_skeleton_geometry_single_point_has_no_lines():
    xy = np.array([[0.0, 0.0]], dtype="float32")
    geo = SkeletonGeometry(xy, edges=[])
    assert geo.lines is None
    # set_thickness should be a no-op, not raise
    geo.set_thickness(5.0)


def test_skeleton_geometry_update(xy3):
    geo = SkeletonGeometry(xy3, edges=[(0, 1), (1, 2)])
    new_xy = xy3 + 1.0
    geo.update(new_xy)
    np.testing.assert_array_equal(geo.points.geometry.positions.data[:, :2], new_xy)
    np.testing.assert_array_equal(
        geo.lines.geometry.positions.data[:, :2], new_xy[geo.edges_idx]
    )


def test_skeleton_geometry_setters(xy3):
    geo = SkeletonGeometry(xy3, edges=[(0, 1)])
    geo.set_color("blue")
    assert geo.points.material.color == gfx.Color("blue")
    geo.set_markersize(20.0)
    assert geo.points.material.size == 20.0
    geo.set_thickness(0.0)
    assert geo.lines.material.opacity == 0
    geo.set_thickness(3.0)
    assert geo.lines.material.opacity == 1
    assert geo.lines.material.thickness == 3.0


def test_skeleton_geometry_state_roundtrip(xy3):
    edges = [(0, 1), (1, 2)]
    geo = SkeletonGeometry(xy3, edges, color="red", markersize=8.0, thickness=2.0)
    state = geo.get_state()
    assert state == {"color": (1.0, 0.0, 0.0, 1.0), "markersize": 8.0, "thickness": 2.0}

    geo2 = SkeletonGeometry.from_state(state, xy3, edges)
    assert geo2.get_state() == state


def test_skeleton_geometry_scene_add_remove(xy3):
    scene = gfx.Scene()
    geo = SkeletonGeometry(xy3, edges=[(0, 1), (1, 2)], scene=scene)
    assert geo.points in scene.children
    assert geo.lines in scene.children

    geo.remove_from_scene()
    assert geo.points not in scene.children
    assert geo.lines not in scene.children
