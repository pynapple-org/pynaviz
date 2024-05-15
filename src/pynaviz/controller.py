"""
	The controller class
"""
from pygfx import PanZoomController,Viewport, Renderer, Camera
from typing import Union, Optional, Callable


class ControllerGroup:

	def __init__(self, *controllers_and_renderers):
		self._controller_group = dict()
		for ctrl_id, cntrl_and_rend in enumerate(controllers_and_renderers):
			ctrl, rend = cntrl_and_rend
			self._controller_group[ctrl_id] = ctrl
			self._add_update_handler(rend)

	def _add_update_handler(self, viewport_or_renderer: Union[Viewport, Renderer]):
		viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
		viewport.renderer.add_event_handler(self.update, "update")

	def add(self, controller):
		pass

	def remove(self, controller_id):
		pass

	def update(self, event):
		print(f"update controller {event['controller_id']}")


class PynaVizController(PanZoomController):
	def __init__(
			self,
			controller_id: int,
			camera: Optional[Camera] = None,
			*,
			enabled=True,
			damping: int = 4,
			auto_update: bool = True,
			register_events: Optional[Union[Viewport, Renderer]] = None,
	):
		self._controller_id = controller_id

		super().__init__(camera=camera, enabled=enabled, damping=damping, auto_update=auto_update, register_events=register_events)
		self.handle_event = None

		if register_events:
			self.handle_event = self._get_event_handle(register_events)

	@staticmethod
	def _get_event_handle(register_events: Union[Viewport, Renderer]) -> Callable:
		"""
		Set up the callback to update.

		:return:
		"""
		# grab the viewport
		viewport = Viewport.from_viewport_or_renderer(register_events)
		return viewport.renderer.handle_event

	def _update_event(self, *args, **kwargs):
		ev = {
			"event_type": "update",
			"controller_id": self._controller_id,
			"args": args,
			"kwargs": kwargs
		}
		if self.handle_event:
			self.handle_event(ev)

	def _update_pan(self, delta, *, vecx, vecy):
		super()._update_pan(delta, vecx=vecx, vecy=vecy)
		self._update_event(vecx=vecx, vecy=vecy)

	def _update_zoom(self, delta):
		super()._update_zoom(delta)
		self._update_event(delta)

	def _update_zoom_to_point(self, delta, *, screen_pos, rect):
		super()._update_zoom_to_point(delta, screen_pos=screen_pos, rect=rect)
		self._update_event(delta, screen_pos=screen_pos, rect=rect)
