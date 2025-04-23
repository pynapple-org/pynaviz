"""
The controller class.
"""

from typing import Callable, Optional, Union

import pygfx
import pygfx as gfx
import pynapple as nap
from debugpy.common.timestamp import current
from pygfx import Camera, PanZoomController, Renderer, Viewport
from pylinalg import vec_transform, vec_unproject

from .events import SyncEvent


def _get_event_handle(renderer: Union[Viewport, Renderer]) -> Callable:
    """
    Set up the callback to update.

    When initializing the custom controller, the method register_events
    is called. It adds to the renderer an event handler by calling
    viewport.renderer.add_event_handler of EventTarget.
    This function grabs the function that loops through the callbacks in
    renderer._event_handlers dictionary.

    :return:
    """
    # grab the viewport
    viewport = Viewport.from_viewport_or_renderer(renderer)
    return viewport.renderer.handle_event


def map_screen_to_world(camera, pos, viewport_size):
    # first convert position to NDC
    x = pos[0] / viewport_size[0] * 2 - 1
    y = -(pos[1] / viewport_size[1] * 2 - 1)
    pos_ndc = (x, y, 0)
    pos_ndc += vec_transform(camera.world.position, camera.camera_matrix)
    # unproject to world space
    pos_world = vec_unproject(pos_ndc[:2], camera.camera_matrix)
    return pos_world


class ControllerGroup:
    """

    Parameters
    ----------
    controllers_and_renderers : list
        controllers and renderers. Can be empty.
    interval : tuple of float or int
        The start and end of the epoch to show when initializing.

    """

    def __init__(self, controllers_and_renderers=[], interval=(0, 10)):
        self._controller_group = dict()
        ids = [
            ctrl.controller_id
            for ctrl, _ in controllers_and_renderers
            if ctrl.controller_id is not None
        ]
        if len(set(ids)) != len(ids):
            raise ValueError("Controller ids must be all different!")
        if ids:
            id0 = max(ids) + 1
        else:
            id0 = 0

        if not isinstance(interval, (tuple, list)):
            raise ValueError("interval should be tuple or list")
        if not len(interval) == 2 and not all(
            [isinstance(x, (float, int)) for x in interval]
        ):
            raise ValueError("interval should be a 2-tuple of float/int")
        if interval[0] > interval[1]:
            raise RuntimeError("interval start should precede interval end")

        self.interval = interval

        for i, cntrl_and_rend in enumerate(controllers_and_renderers):
            ctrl, rend = cntrl_and_rend
            if ctrl.controller_id is None:
                ctrl.controller_id = i + id0
            self._controller_group[ctrl.controller_id] = ctrl
            self._add_update_handler(rend)
            # # Need to move the controllers to show the given interval.
            ctrl.show_interval(*interval)

    def _add_update_handler(self, viewport_or_renderer: Union[Viewport, Renderer]):
        viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
        viewport.renderer.add_event_handler(self.sync_controllers, "sync")

    def add(self, controller, renderer, controller_id):
        if controller_id in self._controller_group.keys():
            raise RuntimeError(
                f"controller_id {controller_id} already in controller group."
            )
        if controller.controller_id is None:
            controller.controller_id = controller_id
        self._controller_group[controller_id] = controller
        self._add_update_handler(renderer)

    def remove(self, controller_id):
        pass

    def sync_controllers(self, event):
        """Sync controllers according to their rule."""
        # print(event)
        for id_other, ctrl in self._controller_group.items():
            if event.controller_id == id_other:
                continue
            ctrl.sync(event)


class CustomController(PanZoomController):
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
        self.camera = (
            camera  # Weirdly pygfx controller doesn't have it as direct attributes
        )
        self.renderer = renderer  # Nor renderer
        self.renderer_handle_event = None
        self.renderer_request_draw = lambda: True

        if renderer:
            self.renderer_handle_event = _get_event_handle(
                renderer
            )  # renderer.handle_event
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
            raise TypeError(
                "When provided, `dict_sync_funcs` must be a dictionary of callables."
            )

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

    def sync(self, event):
        pass

    def show_interval(self, start, end):
        pass


class SpanController(CustomController):
    """
    The class for horizontal time-panning
    """

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
        min=None,
        max=None,
    ):
        super().__init__(
            camera=camera,
            enabled=enabled,
            damping=damping,
            auto_update=auto_update,
            renderer=renderer,
            controller_id=controller_id,
            dict_sync_funcs=dict_sync_funcs,
        )
        self._min = min
        self._max = max
        self.show_interval(0, 1)

    def _update_pan(self, delta, *, vecx, vecy):
        super()._update_pan(delta, vecx=vecx, vecy=vecy)
        self._send_sync_event(
            update_type="pan",
            cam_state=self._get_camera_state(),
            delta=delta,
            vecx=vecx,
            vecy=vecy,
        )

    def _update_zoom(self, delta):
        super()._update_zoom(delta)
        self._send_sync_event(
            update_type="zoom", cam_state=self._get_camera_state(), delta=delta
        )

    def _update_zoom_to_point(self, delta, *, screen_pos, rect):
        super()._update_zoom_to_point(delta, screen_pos=screen_pos, rect=rect)
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
        if "delta_t" in event.kwargs:
            camera_state = self._get_camera_state()
            camera_pos = camera_state["position"].copy()
            camera_pos[0] += event.kwargs["delta_t"]
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
        self.renderer_request_draw()

    def show_interval(self, start, end):
        # # print(start, end, self._min, self._max)
        # viewport_size = self.renderer.logical_size
        #
        # xmin, ymin = 0, self.renderer.logical_size[1]
        # xmax, ymax = self.renderer.logical_size[0], 0
        #
        # world_xmin, world_ymin, _ = map_screen_to_world(self.camera,(xmin, ymin), viewport_size)
        # world_xmax, world_ymax, _ = map_screen_to_world(self.camera,(xmax, ymax), viewport_size)
        #
        # # print(world_xmin, world_xmax, world_ymin, world_ymax)
        # print(self._get_camera_state())

        self.camera.show_rect(  # Uses world coordinates
            left=start, right=end, top=self._min, bottom=self._max
        )
        self._update_cameras()
        self.renderer_request_draw()


class GetController(CustomController):
    """
    The class for grabbing a single time point
    """

    def __init__(
        self,
        camera: Optional[Camera] = None,
        *,
        auto_update: bool = True,
        renderer: Optional[Union[Viewport, Renderer]] = None,
        controller_id: Optional[int] = None,
        data: Optional[Union[nap.TsdFrame, nap.TsdTensor]] = None,
        buffer: pygfx.Buffer = None,
        time_text: gfx.Text = None,
    ):
        super().__init__(
            camera=camera,
            auto_update=auto_update,
            renderer=renderer,
            controller_id=controller_id,
        )
        self.data = data
        self.n_frames = data.shape[0]
        self.frame_index = 0
        self.buffer = buffer
        self.time_text = time_text

    @property
    def frame_index(self):
        return self._frame_index

    @frame_index.setter
    def frame_index(self, value):
        self._frame_index = max(min(value, self.n_frames), 0)

    def _update_zoom_to_point(self, delta, *, screen_pos, rect):
        """Should convert the jump of time to camera position
        before emitting the sync event.
        Does not propagate to the original PanZoomController
        """
        current_t = self.data.index.values[self.frame_index]
        if delta > 0:
            self.frame_index += 1
        else:
            self.frame_index -= 1
        delta_t = self.data.index.values[self.frame_index] - current_t

        if self.buffer.data.shape[0] == 1 and self.buffer.data.shape[1] == 3: # assume single point
            self.buffer.data[0,0:2] = self.data.values[self.frame_index].astype("float32")
        else:
            self.buffer.data[:] = self.data.values[self.frame_index].astype("float32")
        self.buffer.update_full()

        if self.time_text:
            self.time_text.set_text(str(self.data.t[self.frame_index]))

        self.renderer_request_draw()

        # Sending the sync event
        self._send_sync_event(update_type="pan", delta_t=delta_t)

    def sync(self, event):
        """Get a new data point and update the texture"""
        if "cam_state" in event.kwargs:
            new_t = event.kwargs["cam_state"]["position"][0]
        else:
            delta_t = event.kwargs["delta_t"]
            new_t = self.data.t[self.frame_index] + delta_t

        self.frame_index = self.data.get_slice(new_t).start

        if self.buffer.data.shape[0] == 1 and self.buffer.data.shape[1] == 3: # assume single point
            self.buffer.data[0,0:2] = self.data.values[self.frame_index].astype("float32")
        else:
            self.buffer.data[:] = self.data.values[self.frame_index].astype("float32")

        self.buffer.update_full()

        if self.time_text:
            self.time_text.set_text(str(self.data.t[self.frame_index]))

        self.renderer_request_draw()

    def show_interval(self, start, end):
        t = start + (end - start) / 2
        self.frame_index = self.data.get_slice(t).start
        # self.buffer.data[:] = self.data.values[self.frame_index].astype("float32")
        if self.buffer.data.shape[0] == 1 and self.buffer.data.shape[1] == 3: # assume single point
            self.buffer.data[0,0:2] = self.data.values[self.frame_index].astype("float32")
        else:
            self.buffer.data[:] = self.data.values[self.frame_index].astype("float32")

        self.buffer.update_full()
        if self.time_text:
            self.time_text.set_text(str(self.data.t[self.frame_index]))
        self.renderer_request_draw()
