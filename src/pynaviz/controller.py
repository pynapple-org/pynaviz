"""
The controller class.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np
import pygfx
from pygfx import Camera, PanZoomController, Renderer, Viewport

from .events import SwitchEvent, SyncEvent
from .utils import RenderTriggerSource, _get_event_handle


class CustomController(ABC, PanZoomController):
    """"""

    def __init__(
        self,
        camera: Optional[Camera] = None,
        *,
        enabled=True,
        damping: int = 0,
        auto_update: bool = True,
        renderer: Optional[Union[Viewport, Renderer]] = None,
        controller_id: Optional[int] = None,
        dict_sync_funcs: Optional[dict[Callable]] = None,
    ):
        super().__init__(
            camera=camera,
            enabled=enabled,
            damping=damping,
            auto_update=auto_update,
            register_events=renderer,
        )

        if controller_id is not None and not isinstance(controller_id, int):
            raise TypeError(
                f"If provided, `controller_id` must be of integer type. Type {type(controller_id)} provided instead!"
            )
        self._controller_id = controller_id
        self.camera = camera  # Weirdly pygfx controller doesn't have it as direct attributes
        self.renderer = renderer  # Nor renderer
        self.renderer_handle_event = None
        self.renderer_request_draw = lambda: True

        if renderer:
            self.renderer_handle_event = _get_event_handle(renderer)  # renderer.handle_event
            self.renderer_request_draw = lambda: self._request_draw(
                renderer
            )  # renderer.request_draw

        if dict_sync_funcs is None:
            self._dict_sync_funcs = dict()
        elif isinstance(dict_sync_funcs, dict):
            for key, sync_func in dict_sync_funcs.items():
                if not isinstance(sync_func, Callable):
                    raise TypeError(
                        f"`dict_sync_funcs` items must be of `Callable` type. "
                        f"Type {type(sync_func)} for key {key} provided instead!"
                    )
            self._dict_sync_funcs = dict_sync_funcs
        else:
            raise TypeError("When provided, `dict_sync_funcs` must be a dictionary of callables.")

    @property
    def controller_id(self):
        return self._controller_id

    @controller_id.setter
    def controller_id(self, value):
        if self._controller_id is not None:
            raise ValueError("Controller id can be set only once!")
        self._controller_id = value

    def _request_draw(self, viewport):
        if self.auto_update:
            viewport = Viewport.from_viewport_or_renderer(viewport)
            viewport.renderer.request_draw()

    def _send_sync_event(self, update_type: str, *args, **kwargs):
        """
        The function called when moving the objects.
        Passing a pygfx.Event object to the renderer handle_event function.
        It then goes to ControllerGroup to act on the other controllers.
        """
        if self.renderer_handle_event:
            self.renderer_handle_event(
                SyncEvent(
                    type="sync",
                    controller_id=self._controller_id,
                    update_type=update_type,
                    sync_extra_args=dict(args=args, kwargs=kwargs),
                )
            )

    def _send_switch_event(self):
        if self.renderer_handle_event:
            self.renderer_handle_event(
                SwitchEvent(
                    type="switch",
                    controller_id=self._controller_id,
                    new_controller=self,
                    sync_extra_args=dict(args=(), kwargs={}),
                )
            )

    def get_xlim(self):
        """Return the current x boundaries"""
        half_width = self.camera.width / 2
        return self.camera.local.x - half_width, self.camera.local.x + half_width

    @abstractmethod
    def sync(self, event):
        pass

    @abstractmethod
    def advance(self, delta=0.025):
        # This should still trigger a sync event
        pass


class SpanController(CustomController):
    """
    The class for horizontal time-panning
    """

    def __init__(
        self,
        camera: Optional[Camera] = None,
        *,
        enabled: bool = True,
        damping: int = 0,
        auto_update: bool = True,
        renderer: Optional[Union[Viewport, Renderer]] = None,
        controller_id: Optional[int] = None,
        dict_sync_funcs: Optional[dict[Callable]] = None,
        plot_callbacks: Optional[list[Callable]] = None,
    ) -> None:
        super().__init__(
            camera=camera,
            enabled=enabled,
            damping=damping,
            auto_update=auto_update,
            renderer=renderer,
            controller_id=controller_id,
            dict_sync_funcs=dict_sync_funcs,
        )
        self._plot_callbacks = plot_callbacks if plot_callbacks is not None else []

    def set_xlim(self, xmin: float, xmax: float):
        """Set the visible X range for an OrthographicCamera.
        """
        width = xmax - xmin
        x_center = (xmax + xmin) / 2
        self.camera.width = width
        self.camera.local.x = x_center
        self._update_plots()
        self.renderer_request_draw()
        self._send_sync_event(
            update_type="set_xlim",
            cam_state=self._get_camera_state(),
        )

    def set_ylim(self, ymin: float, ymax: float):
        """Set the visible Y range for an OrthographicCamera."""
        height = ymax - ymin
        y_center = (ymax + ymin) / 2
        self.camera.height = height
        self.camera.local.y = y_center
        self._update_plots()
        self.renderer_request_draw()

    def set_view(self, xmin: float, xmax: float, ymin: float, ymax: float):
        """Set the visible X and Y ranges for an OrthographicCamera."""
        self.set_xlim(xmin, xmax)
        self.set_ylim(ymin, ymax)

    def _add_callback(self, func):
        if isinstance(func, Callable):
            self._plot_callbacks.append(func)

    def _update_plots(self):
        for update_func in self._plot_callbacks:
            update_func(**self.camera.get_state())

    def _update_pan(self, delta, *, vecx, vecy):
        super()._update_pan(delta, vecx=vecx, vecy=vecy)
        self._update_plots()
        self._send_sync_event(
            update_type="pan",
            cam_state=self._get_camera_state(),
            delta=delta,
            vecx=vecx,
            vecy=vecy,
        )

    def _update_zoom(self, delta):
        super()._update_zoom(delta)
        self._update_plots()
        self._send_sync_event(
            update_type="zoom", cam_state=self._get_camera_state(), delta=delta
        )

    def _update_zoom_to_point(self, delta, *, screen_pos, rect):
        super()._update_zoom_to_point(delta, screen_pos=screen_pos, rect=rect)
        self._update_plots()
        self._send_sync_event(
            update_type="zoom_to_point",
            cam_state=self._get_camera_state(),
            delta=delta,
            screen_pos=screen_pos,
            rect=rect,
        )

    def sync(self, event):
        """Set a new camera state using the sync rule provided."""
        # Need to convert to camera movement
        if "current_time" in event.kwargs:
            camera_state = self._get_camera_state()
            camera_pos = np.array(camera_state["position"]).copy()
            camera_pos[0] = event.kwargs["current_time"]
            camera_state["position"] = camera_pos
            event.kwargs["cam_state"] = camera_state

        if event.update_type in self._dict_sync_funcs:
            func = self._dict_sync_funcs[event.update_type]
            state_update = func(event, self._get_camera_state())
        else:
            raise NotImplementedError(f"Update {event.update_type} not implemented!")
        # Update camera
        self._set_camera_state(state_update)
        self._update_cameras()
        self._update_plots()
        self.renderer_request_draw()

    def advance(self, delta=0.025):
        """
        Advances the camera's position by a specified delta value along the x-axis.

        This can be used to play the time series with a timer thread.

        Parameters
        ----------
        delta (float): The incremental value to adjust the camera's x-axis position. Defaults to 0.025.

        """
        camera_state = self._get_camera_state()
        new_position = np.array(camera_state["position"]).copy()
        new_position[0] += delta
        # note: self._update_cameras is based on self._last_cam_state.
        # The width of self._last_cam_state can differ from that of camera_state["width"].
        # Provide both position and width for the desired update.
        self._set_camera_state(dict(position=new_position, width=camera_state["width"]))
        self._update_cameras()
        self._update_plots()
        self.renderer_request_draw()
        # To make sure all controller stays in sync
        self._send_sync_event(update_type="pan", current_time=new_position[0])

    def go_to(self, target_time: float):
        """
        Directly set the camera's x-axis position to a specified target time.

        Parameters
        ----------
        target_time (float): The target time to set the camera's x-axis position.

        """
        camera_state = self._get_camera_state()
        new_position = np.array(camera_state["position"]).copy()
        new_position[0] = target_time
        self._set_camera_state(dict(position=new_position))
        self._update_cameras()
        self._update_plots()
        self.renderer_request_draw()
        # To make sure all controller stays in sync
        self._send_sync_event(update_type="pan", current_time=target_time)

class SpanYLockController(SpanController):
    """
    Horizontal time-panning with y-axis locked
    """

    def __init__(self, *args, **kwargs):
        """
        The class for horizontal time-panning and zooming, with the y-axis locked.
        """
        super().__init__(*args, **kwargs)

    def _update_pan(self, delta, *, vecx, vecy):
        """
        Update pan in x axis only, forcing vecy to be 0.
        """
        super()._update_pan(delta, vecx=vecx, vecy=0)

    def _update_zoom(self, delta):
        """
        Rewrite of _update_zoom since its inputs don't allow separation of fx and fy
        """
        if isinstance(delta, (int, float)):
            delta = (delta, delta)
        assert isinstance(delta, tuple) and len(delta) == 2

        fx = 2 ** delta[0]
        new_cam_state = self._zoom(fx, 1, self._get_camera_state())
        self._set_camera_state(new_cam_state)
        self._send_sync_event(
            update_type="zoom", cam_state=self._get_camera_state(), delta=delta
        )

    def _zoom(self, fx, fy, cam_state):
        """
        Zoom in x axis only, enforcing fy to be 1.
        """
        return super()._zoom(fx, 1, cam_state)


class GetController(CustomController):
    """
    The class for grabbing a single time point
    """

    def __init__(
        self,
        camera: Optional[Camera] = None,
        *,
        enabled=True,
        auto_update: bool = True,
        renderer: Optional[Union[Viewport, Renderer]] = None,
        controller_id: Optional[int] = None,
        data: Optional[Any] = None,
        buffer: pygfx.Buffer = None,
        callback: Optional[Callable] = None,
    ):
        super().__init__(
            camera=camera,
            enabled=enabled,
            auto_update=auto_update,
            renderer=renderer,
            controller_id=controller_id,
        )
        self.data = data
        if self.data:
            self.frame_index = 0
            self._current_time = self._get_frame_time()  # Initializing the current time
        else:
            self._current_time = None

        self.buffer = buffer
        self._plot_callbacks = [callback] if callback is not None else []

    def set_view(self, xmin: float, xmax: float, ymin: float, ymax: float):
        """Set the visible X and Y ranges for an OrthographicCamera."""
        if self.camera is not None:
            self.camera.show_rect(xmin, xmax, ymin, ymax)

    @property
    def frame_index(self):
        return self._frame_index

    @frame_index.setter
    def frame_index(self, value):
        if self.data:
            n_frames = self.data.shape[0]
            self._frame_index = max(min(value, n_frames), 0)
        else:
            self._frame_index = 0

    def _get_frame_time(self):
        time_array = getattr(self.data.index, "values", self.data.index)
        return time_array[self.frame_index]

    def _add_callback(self, func):
        if isinstance(func, Callable):
            self._plot_callbacks.append(func)

    def _update_buffer(self, event_type: Optional[RenderTriggerSource] = None):
        for update_func in  self._plot_callbacks:
            update_func(self.frame_index, event_type)

    def _update_zoom_to_point(self, delta, *, screen_pos, rect):
        """Should convert the jump of time to camera position
        before emitting the sync event.
        Does not propagate to the original PanZoomController
        """
        if delta > 0:
            self.frame_index += 1
        else:
            self.frame_index -= 1

        self.frame_index = min(max(self.frame_index, 0), self.data.shape[0] - 1)

        self._update_buffer(event_type=RenderTriggerSource.ZOOM_TO_POINT)

        # hack for finding out if data is pynapple
        # TODO fix later
        if hasattr(self.data.index, "values"):
            # Sending the sync event (no concurrent logic)
            self._current_time = self._get_frame_time()
            self._send_sync_event(update_type="pan", current_time=self._current_time)

    def set_frame(self, target_time: float):
        """
        Set the frame from target time.

        Parameters
        ----------
        target_time:
            A time point.
        """
        time_array = getattr(self.data.index, "values", self.data.index)
        idx_before = np.searchsorted(time_array, target_time, side="right") - 1
        idx_before = np.clip(idx_before, 0, len(time_array) - 1)
        idx_after = min(idx_before + 1, len(time_array) - 1)
        frame_index = (
            idx_before
            if (time_array[idx_after] - target_time) > (target_time - time_array[idx_before])
            else idx_after
        )
        current_t = time_array[frame_index]

        # update frame index
        self.frame_index = frame_index

        # update buffer and sync
        self._current_time = target_time  # Target time is not necessarily frame time
        self._update_buffer(event_type=RenderTriggerSource.SET_FRAME)
        self._send_sync_event(update_type="pan", current_time=current_t)


    def sync(self, event):
        """Get a new data point and update the texture"""
        new_t = None
        if "cam_state" in event.kwargs:
            new_t = event.kwargs["cam_state"]["position"][0]
            self._current_time = new_t
        elif "current_time" in event.kwargs:
            self._current_time = event.kwargs["current_time"]
            index = np.searchsorted(self.data.index, self._current_time, side="right") - 1
            index = np.clip(index, 0, len(self.data.index) - 1)
            new_t = self.data.t[index]

        if new_t is not None:
            self.frame_index = self.data.get_slice(new_t).start
            self._update_buffer(
                RenderTriggerSource.SYNC_EVENT_RECEIVED
            )  # self.buffer.data[:] = self.data.values[self.frame_index].astype("float32")

    def advance(self, delta=0.025):
        """
        Advance the current time by a specified delta value.

        This can be used to play movies with a timer thread.

        Parameters
        ----------
        delta (float): The incremental value. Defaults to 0.025.

        """
        self._current_time += delta
        # set frame sends a sync
        self.set_frame(self._current_time)
