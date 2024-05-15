"""
	The controller class
"""
from pygfx import PanZoomController,Viewport, Renderer, Camera
from typing import Union, Optional, Callable



class ControllerGroup:

	def __init__(self):
		pass

	def add(self, controller):
		pass

	def remove(self, controller_id):
		pass

	def update(self, event, *args, **kwargs):
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
			update_group: Optional[Callable] = None,
	):
		self._controller_id = controller_id

		super().__init__(camera=camera, enabled=enabled, damping=damping, auto_update=auto_update, register_events=register_events)

		self.handle_event = None
		if register_events is not None and update_group is not None:
			self.handle_event = self._set_up_update_callback(register_events, update_group)



	@staticmethod
	def _set_up_update_callback(register_events: Union[Viewport, Renderer], update_group: Callable) -> Callable:
		"""
		Set up the callback to update.

		:return:
		"""
		# grab the viewport
		viewport = Viewport.from_viewport_or_renderer(register_events)
		# set up the update callback
		viewport.renderer.add_event_handler(update_group, "update")
		return viewport.renderer._handle_event_and_flush

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
