"""
ControllerGroup is used to synchronize in time each canvas.
"""

from typing import Callable, Optional, Sequence, Union

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
    callback : Optional[Callable]
        A function to be called when the time is advanced through any sync event.
        This is used mostly for Qtimer integration in GUI applications.
    """

    def __init__(
        self,
        plots: Optional[Sequence] = None,
        interval: tuple[Union[int, float], Union[int, float]] = (0, 1),
        callback: Optional[Callable] = None,
    ):
        self._controller_group = dict()
        self.callback = callback
        self.current_time = None
        self.interval = interval

        # Validate interval format
        if not isinstance(interval, (tuple, list)):
            raise ValueError("`interval` must be a tuple or list.")

        if len(interval) != 2 or not all(isinstance(x, (int, float)) for x in interval):
            raise ValueError("`interval` must be a 2-tuple of int or float values.")

        # Initialize controller group from given plots
        if plots is not None:
            for i, plt in enumerate(plots):
                plt.controller._controller_id = i
                self._add_update_handler(plt.renderer)
                self._controller_group[i] = plt.controller

        self.set_interval(interval[0], interval[1])


    def _add_update_handler(self, viewport_or_renderer: Union[Viewport, Renderer]):
        """
        Registers a sync event handler on the renderer of the given viewport or renderer.
        """
        viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
        viewport.renderer.add_event_handler(self.sync_controllers, "sync")
        viewport.renderer.add_event_handler(self.switch_controller, "switch")

    def set_interval(self, start: Union[int, float], end: Union[int, float, None]):
        """
        Sets a new time interval for all controllers in the group.

        Parameters
        ----------
        start : int or float
            The start of the new time interval.
        end : int or float or None
            The end of the new time interval. If None, go to start.

        Raises
        ------
        ValueError
            If start is greater than end.
        """
        if end is not None and start > end:
            raise ValueError("`start` must not be greater than `end`.")

        self.interval = (start, end)
        if end is None:
            self._set_to_time(start)
            self.current_time = start
        else:
            self._set_from_start_end(start, end)
            self.current_time = start + (end - start) / 2

        # Call the callback if provided
        if self.callback is not None:
            self.callback(self.current_time)

    def _set_to_time(self, time):
        for ctrl in self._controller_group.values():
            if not ctrl.enabled:
                continue
            ctrl.sync(
                SyncEvent(
                    type="sync",
                    controller_id=ctrl.controller_id,
                    update_type="pan",
                    sync_extra_args={
                        "args": (),
                        "kwargs": {
                            "current_time": time
                        }
                    }
                )
            )

    def _set_from_start_end(self, start, end):
        for ctrl in self._controller_group.values():
            if not ctrl.enabled:
                continue
            if hasattr(ctrl, "set_xlim"):
                ctrl.set_xlim(start, end)
                break
            else:
                self._set_to_time(start + (end - start) / 2)
                break



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

        # Call the callback if provided
        if self.callback is not None:
            self.callback(self.current_time)

    def switch_controller(self, event):
        """
        Switches the active controller in the group based on a switch event.

        Parameters
        ----------
        event : Event
            The switch event that contains `controller_id` to switch.
        """
        if not hasattr(event, "new_controller") or event.new_controller is None:
            return
        else:
            ctrl = self._controller_group.get(event.controller_id, None)
            if ctrl is not None and event.controller_id in self._controller_group:
                self._controller_group[event.controller_id] = event.new_controller

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

        # set the private method to avoid checks
        controller._controller_id = controller_id

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
        except Exception as e:
            # Fallback: skip if removal fails (e.g., missing references)
            print(f"Failed to remove event handle with exception:\n{e}")

        try:
            viewport = Viewport.from_viewport_or_renderer(controller.renderer)
            viewport.renderer.remove_event_handler(self.switch_controller, "switch")
        except Exception as e:
            # Fallback: skip if removal fails (e.g., missing references)
            print(f"Failed to remove event handle with exception:\n{e}")

