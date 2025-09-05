"""
ControllerGroup is used to synchronize in time each canvas.
"""

from typing import Optional, Sequence, Union

import numpy as np
from pygfx import Renderer, Viewport

from pynaviz.events import SyncEvent


class ControllerGroup:
    """
    Manages a group of plot controllers and synchronizes them.

    Parameters
    ----------
    plots : Optional[Sequence]
        A sequence of plot objects (each with a `.controller` and `.renderer` attribute).
        Can be None or empty.
    interval : tuple[float | int, float | int]
        Start and end of the epoch (x-axis range) to show when initializing.
        Must be a 2-tuple with start <= end.
    """

    def __init__(
        self,
        plots: Optional[Sequence] = None,
        interval: tuple[Union[int, float], Union[int, float]] = (0, 1),
    ):
        self._controller_group = dict()

        # Validate interval format
        if not isinstance(interval, (tuple, list)):
            raise ValueError("`interval` must be a tuple or list.")

        if len(interval) != 2 or not all(isinstance(x, (int, float)) for x in interval):
            raise ValueError("`interval` must be a 2-tuple of int or float values.")

        if interval[0] > interval[1]:
            raise ValueError("`interval` start must not be greater than end.")

        self.interval = interval
        self.current_time = interval[0] + (interval[1] - interval[0])/2

        # Initialize controller group from given plots
        if plots is not None:
            for i, plt in enumerate(plots):
                plt.controller._controller_id = i
                self._add_update_handler(plt.renderer)
                plt.controller.set_xlim(*interval)
                self._controller_group[i] = plt.controller

    def _add_update_handler(self, viewport_or_renderer: Union[Viewport, Renderer]):
        """
        Registers a sync event handler on the renderer of the given viewport or renderer.
        """
        viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
        viewport.renderer.add_event_handler(self.sync_controllers, "sync")

    def sync_controllers(self, event):
        """
        Synchronizes all other controllers in the group when a sync event is triggered.

        Parameters
        ----------
        event : Event
            The sync event that contains `controller_id` and possibly data to sync.
        """
        # Intercepting the new current time
        if hasattr(event, "kwargs") and "cam_state" in event.kwargs:
            self.current_time = event.kwargs["cam_state"]["position"][0]
        else:
            self.current_time = event.kwargs["current_time"]

        for id_other, ctrl in self._controller_group.items():
            if event.controller_id != id_other and ctrl.enabled:
                ctrl.sync(event)

    def advance(self, delta=0.025):
        """
        Advances the simulation or application by a specified time interval.

        This method updates the current simulation or application time and triggers
        synchronized events for all enabled controllers within the controller group.
        The method generates a sync event with specific parameters and uses this event
        to synchronize each enabled controller.

        Parameters
        ----------
        delta (float): The time interval to advance the simulation or application.
                       Defaults to 0.025 seconds.

        """
        try:
            first_key = next(iter(self._controller_group))
            self._controller_group[first_key].advance(delta)
            self.current_time += delta
        except StopIteration:
            # Dictionary is empty, nothing to advance
            print("Controller group is empty, nothing to advance.")

    def add(self, plot, controller_id: int):
        """
        Adds a plot to the controller group.

        Parameters
        ----------
        plot : object
            A base or widget plot with a `.controller` and `.renderer` attribute,
            or a wrapper with `.plot.controller` and `.plot.renderer`.
        controller_id : int
            Unique identifier to assign to the plot's controller in the group.

        Raises
        ------
        RuntimeError
            If the plot doesn't have a controller/renderer or the ID already exists.
        """
        # Attempt to extract controller and renderer
        if hasattr(plot, "controller") and hasattr(plot, "renderer"):
            controller = plot.controller
            renderer = plot.renderer
        elif (
            hasattr(plot, "plot")
            and hasattr(plot.plot, "controller")
            and hasattr(plot.plot, "renderer")
        ):
            controller = plot.plot.controller
            renderer = plot.plot.renderer
        else:
            raise RuntimeError("Plot object must have a controller and renderer.")

        # Prevent duplicate controller IDs
        if controller_id in self._controller_group:
            raise RuntimeError(f"Controller ID {controller_id} already exists in the group.")

        # Assign ID if not already assigned
        if controller.controller_id is None:
            controller.controller_id = controller_id

        self._controller_group[controller_id] = controller
        self._add_update_handler(renderer)

        # Sync the new visual to the current time
        controller.sync(
            SyncEvent(
                type="sync",
                controller_id=controller.controller_id,
                update_type="pan",
                sync_extra_args={
                    "args": (),
                    "kwargs": {
                        "current_time": self.current_time
                    }
                }
            )
        )

    def remove(self, controller_id: int):
        """
        Removes a controller from the group by its ID.

        Parameters
        ----------
        controller_id : int
            The ID of the controller to remove.

        Raises
        ------
        KeyError
            If the controller_id is not found in the group.
        """
        if controller_id not in self._controller_group:
            raise KeyError(f"Controller ID {controller_id} not found in the group.")

        controller = self._controller_group.pop(controller_id)

        # Optional: remove event handler if needed
        # This assumes controller has a reference to its renderer
        try:
            viewport = Viewport.from_viewport_or_renderer(controller.renderer)
            viewport.renderer.remove_event_handler(self.sync_controllers, "sync")
        except Exception:
            # Fallback: skip if removal fails (e.g., missing references)
            pass

