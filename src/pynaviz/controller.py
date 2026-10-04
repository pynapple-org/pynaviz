"""
The controller class.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import pygfx
from pygfx import Camera, PanZoomController, Renderer, Viewport

from .events import SwitchEvent, SyncEvent
from .utils import RenderTriggerSource, _get_event_handle


class CustomController(ABC, PanZoomController):
    def __init__(
        self,
        camera: Camera | None = None,
        *,
        enabled=True,
        damping: int = 0,
        auto_update: bool = True,
        renderer: Viewport | Renderer | None = None,
        controller_id: int | None = None,
        dict_sync_funcs: dict[Callable] | None = None,
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
            self._dict_sync_funcs = {}
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
                    sync_extra_args={"args": args, "kwargs": kwargs},
                )
            )

    def _send_switch_event(self):
        if self.renderer_handle_event:
            self.renderer_handle_event(
                SwitchEvent(
                    type="switch",
                    controller_id=self._controller_id,
                    new_controller=self,
                    sync_extra_args={"args": (), "kwargs": {}},
                )
            )

    def get_xlim(self):
        """Return the current x boundaries"""
        half_width = self.camera.width / 2
        return self.camera.local.x - half_width, self.camera.local.x + half_width

    def get_ylim(self):
        """Return the current y boundaries"""
        half_height = self.camera.height / 2
        return self.camera.local.y - half_height, self.camera.local.y + half_height

    def get_view(self) -> tuple[float, float, float, float]:
        """Return the current visible range as (xmin, xmax, ymin, ymax).

        Camera internals use float32, so values are normalised through float32
        to keep ``get_view`` stable across a ``set_view`` → ``get_view`` roundtrip.
        """
        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()
        return tuple(float(np.float32(v)) for v in (xmin, xmax, ymin, ymax))

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
        camera: Camera | None = None,
        *,
        enabled: bool = True,
        damping: int = 0,
        auto_update: bool = True,
        renderer: Viewport | Renderer | None = None,
        controller_id: int | None = None,
        dict_sync_funcs: dict[Callable] | None = None,
        plot_callbacks: list[Callable] | None = None,
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
        """Set the visible X range for an OrthographicCamera."""
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
        self._send_sync_event(update_type="zoom", cam_state=self._get_camera_state(), delta=delta)

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
        self._set_camera_state({"position": new_position, "width": camera_state["width"]})
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
        self._set_camera_state({"position": new_position})
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
        self._send_sync_event(update_type="zoom", cam_state=self._get_camera_state(), delta=delta)

    def _zoom(self, fx, fy, cam_state):
        """
        Zoom in x axis only, enforcing fy to be 1.
        """
        return super()._zoom(fx, 1, cam_state)


class SpanXLockController(SpanController):
    """
    Vertical panning with x-axis locked (prevents horizontal pan/zoom).
    """

    def _update_pan(self, delta, *, vecx, vecy):
        """Update pan in y axis only, forcing vecx to be 0."""
        super()._update_pan(delta, vecx=0, vecy=vecy)

    def _update_zoom(self, delta):
        """Zoom in y axis only, enforcing fx to be 1."""
        if isinstance(delta, (int, float)):
            delta = (delta, delta)
        assert isinstance(delta, tuple) and len(delta) == 2

        fy = 2 ** delta[1]
        new_cam_state = self._zoom(1, fy, self._get_camera_state())
        self._set_camera_state(new_cam_state)
        self._send_sync_event(update_type="zoom", cam_state=self._get_camera_state(), delta=delta)

    def _zoom(self, fx, fy, cam_state):
        """Zoom in y axis only, enforcing fx to be 1."""
        return super()._zoom(1, fy, cam_state)


class SpanXYLockController(SpanController):
    """
    Both axes locked — no manual pan or zoom. Playback (advance) still works.
    """

    def _update_pan(self, delta, *, vecx, vecy):
        pass  # no-op

    def _update_zoom(self, delta):
        pass  # no-op


class GetController(CustomController):
    """Controller for selecting a single time point."""

    def __init__(
        self,
        camera: Camera | None = None,
        *,
        enabled: bool = True,
        auto_update: bool = True,
        renderer: Viewport | Renderer | None = None,
        controller_id: int | None = None,
        data: Any | None = None,
        buffer: pygfx.Buffer | None = None,
        plot_callbacks: list[Callable] | None = None,
        continuous_time: bool = False,
    ) -> None:
        super().__init__(
            camera=camera,
            enabled=enabled,
            auto_update=auto_update,
            renderer=renderer,
            controller_id=controller_id,
        )
        self.data = data
        self.buffer = buffer
        self.continuous_time = continuous_time
        self._plot_callbacks = list(plot_callbacks) if plot_callbacks is not None else []
        self._frame_index = 0
        self._current_time = None

        if self.data is not None and len(self.data):
            self._current_time = self._get_frame_time()

    def set_view(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> None:
        """Set the visible X and Y ranges."""
        if self.camera is not None:
            self.camera.show_rect(xmin, xmax, ymin, ymax)

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @frame_index.setter
    def frame_index(self, value: int) -> None:
        if self.data is None or not len(self.data):
            self._frame_index = 0
            return

        self._frame_index = int(np.clip(value, 0, len(self.data) - 1))

    def _get_time_array(self) -> np.ndarray:
        """Return data timestamps as an array."""
        if self.data is None:
            return np.array([], dtype=float)

        if hasattr(self.data, "t"):
            return np.asarray(self.data.t)

        index = self.data.index
        return np.asarray(getattr(index, "values", index))

    def _get_frame_time(self) -> float:
        """Return the timestamp of the selected frame."""
        timestamps = self._get_time_array()
        return float(timestamps[self.frame_index])

    @staticmethod
    def _nearest_frame_index(
        timestamps: np.ndarray,
        target_time: float,
    ) -> int:
        """Return the frame nearest to a target time."""
        right = int(np.searchsorted(timestamps, target_time))

        if right <= 0:
            return 0
        if right >= len(timestamps):
            return len(timestamps) - 1

        left = right - 1
        if timestamps[right] - target_time > target_time - timestamps[left]:
            return left

        return right

    def _add_callback(self, func: Callable) -> None:
        if isinstance(func, Callable):
            self._plot_callbacks.append(func)

    def _update_buffer(
        self,
        event_type: RenderTriggerSource | None = None,
    ) -> None:
        for update_func in self._plot_callbacks:
            update_func(self.frame_index, event_type)

    def _update_zoom_to_point(
        self,
        delta,
        *,
        screen_pos,
        rect,
    ) -> None:
        """Move forward or backward by one frame."""
        if self.data is None or not len(self.data):
            return

        self.frame_index += 1 if delta > 0 else -1
        self._current_time = self._get_frame_time()

        self._update_buffer(event_type=RenderTriggerSource.ZOOM_TO_POINT)
        self.renderer_request_draw()
        self._send_sync_event(
            update_type="pan",
            current_time=self._current_time,
        )

    def set_frame(self, target_time: float) -> None:
        """Select the frame nearest to a target time."""
        timestamps = self._get_time_array()
        if not len(timestamps):
            return

        target_time = float(np.clip(target_time, timestamps[0], timestamps[-1]))
        self.frame_index = self._nearest_frame_index(
            timestamps,
            target_time,
        )

        # Store continuous time before callbacks are invoked.
        self._current_time = target_time
        self._update_buffer(event_type=RenderTriggerSource.SET_FRAME)
        self.renderer_request_draw()

        # Existing users synchronize to real frame timestamps. Continuous
        # views synchronize to the unquantized playhead time.
        sync_time = target_time if self.continuous_time else float(timestamps[self.frame_index])
        self._send_sync_event(
            update_type="pan",
            current_time=sync_time,
        )

    def sync(self, event) -> None:
        """Select a frame from a synchronization event."""
        if self.data is None or not len(self.data):
            return

        if "cam_state" in event.kwargs:
            target_time = float(event.kwargs["cam_state"]["position"][0])
        elif "current_time" in event.kwargs:
            target_time = float(event.kwargs["current_time"])
        else:
            return

        timestamps = self._get_time_array()
        target_time = float(np.clip(target_time, timestamps[0], timestamps[-1]))

        # Preserve the previous sync behavior: select the frame at or
        # immediately before the synchronized time.
        frame_index = (
            np.searchsorted(
                timestamps,
                target_time,
                side="right",
            )
            - 1
        )
        self.frame_index = int(np.clip(frame_index, 0, len(timestamps) - 1))
        self._current_time = target_time

        self._update_buffer(RenderTriggerSource.SYNC_EVENT_RECEIVED)
        self.renderer_request_draw()

    def advance(self, delta: float = 0.025) -> None:
        """Advance the playhead by a time delta."""
        if self.data is None or not len(self.data):
            return

        if self._current_time is None:
            self._current_time = self._get_frame_time()

        self.set_frame(self._current_time + delta)
