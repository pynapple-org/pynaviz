import itertools

import numpy as np
import pygfx as gfx


class PlotPoints:
    """
    A streaming skeleton with pygfx Points and Lines.

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
    """
    def __init__(self, tsdframe, initial_time, scene, color = "red", markersize=8.0, thickness=2.0):
        self.data = tsdframe
        self.n_points = self.data.shape[1] // 2

        current_xy = self.data.get(initial_time)
        xy = np.hstack((current_xy.reshape(self.n_points, 2), np.ones((self.n_points, 1)))).astype("float32")

        # ---- Points ----
        geom_points = gfx.Geometry(positions=xy)
        mat_points = gfx.PointsMaterial(color=color, size=markersize)
        self.points = gfx.Points(geom_points, mat_points)
        scene.add(self.points)

        # # ---- Lines ----
        if self.n_points:
            edges = list(itertools.combinations(range(self.n_points), 2))
            self.n_lines = len(edges)
            self.edges_idx = np.array(edges).flatten()

            line_positions = xy[self.edges_idx].astype("float32")


            geom_lines = gfx.Geometry(positions=line_positions)
            mat_lines = gfx.LineMaterial(thickness=thickness,
                                         color="grey")
            self.lines = gfx.Line(geom_lines, mat_lines)

            scene.add(self.lines)


    def update(self, t):
        """
        Update skeleton to positions at time t.

        Parameters
        ----------
        t : float
            Time to display.
        """
        current_xy = self.data.get(t)
        xy = np.hstack((current_xy.reshape(self.n_points, 2), np.ones((self.n_points, 1)))).astype("float32")
        self.points.geometry.positions.set_data(xy)
        self.lines.geometry.positions.set_data(xy[self.edges_idx])

    def set_color(self, color):
        """
        Set color of the points.

        Parameters
        ----------
        color : tuple of float or str
            Color of the points as RGBA values between 0 and 1 or string.
        """
        self.points.material.color = color

    def set_markersize(self, markersize):
        """
        Set size of the points.

        Parameters
        ----------
        markersize : float
            Size of the points in pixels.
        """
        self.points.material.size = markersize

    def set_thickness(self, thickness):
        """
        Set thickness of the connecting lines.

        Parameters
        ----------
        thickness : float
            Thickness of connecting lines.
        """
        if thickness <= 1e-3:
            self.lines.material.opacity = 0
        else:
            self.lines.material.opacity = 1
            self.lines.material.thickness = thickness
