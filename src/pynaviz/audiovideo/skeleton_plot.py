from ..skeleton_geometry import (
    SkeletonGeometry,
    edges_from_parent_metadata,
    resolve_edges,
    resolve_keypoint_labels,
)


class PlotPoints:
    """
    A streaming skeleton with pygfx Points and Lines.

    Thin wrapper around :class:`pynaviz.skeleton_geometry.SkeletonGeometry` that
    adapts it to the video-overlay call pattern (driven by ``update(t)`` on a
    ``nap.TsdFrame``). When ``edges`` is not given, falls back to the data's
    ``"parent"`` column metadata if present (see
    :func:`pynaviz.skeleton_geometry.edges_from_parent_metadata`), otherwise
    every keypoint is connected to every other (complete graph).

    Parameters
    ----------
    tsdframe : nap.TsdFrame
        Skeleton time series data to overlay.
    initial_time : float
        Initial time to display.
    scene : gfx.Scene
        The scene to add the skeleton to.
    color : tuple of float or str
        Color of the points as RGBA values between 0 and 1 or string.
    markersize : float
        Size of the points in pixels.
    thickness : float
        Thickness of connecting lines.
    edges : list of (str or int, str or int), optional
        Bone connectivity as pairs of keypoint labels or indices.
    """
    def __init__(self, tsdframe, initial_time, scene, color = "red", markersize=8.0, thickness=0.02, edges=None):
        self.data = tsdframe
        self.n_points = self.data.shape[1] // 2
        self.labels = resolve_keypoint_labels(self.data.columns)
        self.edges = edges

        current_xy = self.data.get(initial_time).reshape(self.n_points, 2)
        edges_ = self.edges
        if edges_ is None:
            edges_ = edges_from_parent_metadata(self.data)
        edges_idx = resolve_edges(self.labels, edges_)

        self._geometry = SkeletonGeometry(
            current_xy, edges_idx, scene=scene, color=color, markersize=markersize, thickness=thickness
        )
        self.points = self._geometry.points
        if self._geometry.lines is not None:
            self.lines = self._geometry.lines

    def update(self, t):
        """
        Update skeleton to positions at time t.

        Parameters
        ----------
        t : float
            Time to display.
        """
        current_xy = self.data.get(t).reshape(self.n_points, 2)
        self._geometry.update(current_xy)

    def set_color(self, color):
        """
        Set color of the points.

        Parameters
        ----------
        color : tuple of float or str
            Color of the points as RGBA values between 0 and 1 or string.
        """
        self._geometry.set_color(color)

    def set_markersize(self, markersize):
        """
        Set size of the points.

        Parameters
        ----------
        markersize : float
            Size of the points in pixels.
        """
        self._geometry.set_markersize(markersize)

    def set_thickness(self, thickness):
        """
        Set thickness of the connecting lines.

        Parameters
        ----------
        thickness : float
            Thickness of connecting lines.
        """
        self._geometry.set_thickness(thickness)

    def get_state(self) -> dict:
        state = self._geometry.get_state()
        if self.edges is not None:
            state["edges"] = self.edges
        return state

    @classmethod
    def from_state(cls, state, scene, tsdframe, initial_time: float=0.):
        return cls(tsdframe, initial_time=initial_time, scene=scene, **state)


