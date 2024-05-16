"""
	The controller class.
"""

from typing import Callable, Optional, Union

from pygfx import Camera, PanZoomController, Renderer, Viewport

from .events import SyncEvent


class ControllerGroup:

	def __init__(self, *controllers_and_renderers):
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

		for i, cntrl_and_rend in enumerate(controllers_and_renderers):
			ctrl, rend = cntrl_and_rend
			if ctrl.controller_id is None:
				ctrl.controller_id = i + id0
			self._controller_group[ctrl.controller_id] = ctrl
			self._add_update_handler(rend)

	def _add_update_handler(self, viewport_or_renderer: Union[Viewport, Renderer]):
		viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
		viewport.renderer.add_event_handler(self.sync_controllers, "sync")

	def add(self, controller):
		pass

	def remove(self, controller_id):
		pass

	def sync_controllers(self, event):
		"""Sync controllers according to their rule."""
		for id_other, ctrl in self._controller_group.items():
			if event.controller_id == id_other:
				continue
			ctrl.sync(event)


class PynaVizController(PanZoomController):
	def __init__(
		self,
		camera: Optional[Camera] = None,
		*,
		enabled=True,
		damping: int = 4,
		auto_update: bool = True,
		register_events: Optional[Union[Viewport, Renderer]] = None,
		controller_id: Optional[int] = None,
		dict_sync_funcs: Optional[dict[Callable]] = None,
	):

		self._controller_id = controller_id

		super().__init__(
			camera=camera,
			enabled=enabled,
			damping=damping,
			auto_update=auto_update,
			register_events=register_events,
		)
		self.renderer_handle_event = None
		self._draw = lambda: True

		if register_events:
			self.renderer_handle_event = self._get_event_handle(register_events)
			self._draw = lambda: self._request_draw(register_events)

		if dict_sync_funcs is None:
			self._dict_sync_funcs = dict()
		else:
			self._dict_sync_funcs = dict_sync_funcs

	@property
	def controller_id(self):
		return self._controller_id

	@controller_id.setter
	def controller_id(self, value):
		if self._controller_id is not None:
			raise ValueError("Controller id can be set only once!")
		self._controller_id = value

	@staticmethod
	def _get_event_handle(register_events: Union[Viewport, Renderer]) -> Callable:
		"""
		Set up the callback to update.

		:return:
		"""
		# grab the viewport
		viewport = Viewport.from_viewport_or_renderer(register_events)
		return viewport.renderer.handle_event

	def _request_draw(self, viewport):
		if self.auto_update:
			viewport = Viewport.from_viewport_or_renderer(viewport)
			viewport.renderer.request_draw()

	def _update_event(self, update_type: str, *args, **kwargs):
		if self.renderer_handle_event:
			self.renderer_handle_event(
				SyncEvent(
					"sync",
					controller_id=self._controller_id,
					update_type=update_type,
					sync_extra_args=dict(args=args, kwargs=kwargs),
				)
			)

	def _update_pan(self, delta, *, vecx, vecy):
		super()._update_pan(delta, vecx=vecx, vecy=vecy)
		self._update_event(
			update_type="pan",
			cam_state=self._get_camera_state(),
			delta=delta,
			vecx=vecx,
			vecy=vecy,
		)

	def _update_zoom(self, delta):
		super()._update_zoom(delta)
		self._update_event(
			update_type="zoom", cam_state=self._get_camera_state(), delta=delta
		)

	def _update_zoom_to_point(self, delta, *, screen_pos, rect):
		super()._update_zoom_to_point(delta, screen_pos=screen_pos, rect=rect)
		self._update_event(
			update_type="zoom_to_point",
			cam_state=self._get_camera_state(),
			delta=delta,
			screen_pos=screen_pos,
			rect=rect,
		)

	def sync(self, event):
		"""Set a new camera state using the sync rule provided."""
		if event.update_type in self._dict_sync_funcs:
			func = self._dict_sync_funcs[event.update_type]
			state_update = func(event, self._get_camera_state())
		else:
			raise NotImplemented(f"Update {event.update_type} not implemented!")
		# Update camera
		self._set_camera_state(state_update)
		self._update_cameras()
		self._draw()
