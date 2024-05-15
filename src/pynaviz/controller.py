"""
	The controller class
"""
from pygfx import PanZoomController,Viewport, Renderer, Camera, Event
from typing import Union, Optional, Callable
from pygfx.cameras._perspective import fov_distance_factor
import pylinalg as la


class SyncEvent(Event):
	"""Keyboard button press.

	Parameters
	----------
	args :  Any
		Positional arguments are forwarded to the :class:`base class
		<pygfx.objects.Event>`.
	key : str
		The key that was pressed.
	modifiers : list
		The modifiers that were pressed while the key was pressed.
	kwargs : Any
		Additional keyword arguments are forward to the :class:`base class
		<pygfx.objects.Event>`.

	"""
	def __init__(self, *args, controller_id=None, data=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.controller_id = controller_id
		self.data = data

class SetStateAndDrawEvent(Event):
	"""Keyboard button press.

	Parameters
	----------
	args :  Any
		Positional arguments are forwarded to the :class:`base class
		<pygfx.objects.Event>`.
	key : str
		The key that was pressed.
	modifiers : list
		The modifiers that were pressed while the key was pressed.
	kwargs : Any
		Additional keyword arguments are forward to the :class:`base class
		<pygfx.objects.Event>`.

	"""
	def __init__(self, *args, controller_id=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.controller_id = controller_id



class ControllerGroup:

	def __init__(self, *controllers_and_renderers):
		self._controller_group = dict()
		ids = [cntrl._controller_id for cntrl, _ in controllers_and_renderers if cntrl._controller_id is not None]
		if len(set(ids)) != len(ids):
			raise ValueError("Controller ids must be all different!")
		if ids:
			id0 = max(ids) + 1
		else:
			id0 = 0

		for i, cntrl_and_rend in enumerate(controllers_and_renderers):
			ctrl, rend = cntrl_and_rend
			if ctrl._controller_id is None:
				ctrl._controller_id = i + id0
			self._controller_group[ctrl._controller_id] = ctrl
			self._add_update_handler(rend)

	def _add_update_handler(self, viewport_or_renderer: Union[Viewport, Renderer]):
		viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
		viewport.renderer.add_event_handler(self.update, "update")
		viewport.renderer.add_event_handler(self.update, "set_and_draw")


	def add(self, controller):
		pass

	def remove(self, controller_id):
		pass

	def update(self, event):
		# print(f"update controller {event.controller_id}")
		update_type = event.data["kwargs"]["update_type"]
		for id_other, ctrl in self._controller_group.items():
			if event.controller_id == id_other:
				continue
			if update_type == "pan":
				ctrl.compensate_pan(*event.data["args"], **event.data["kwargs"])
			elif update_type == "zoom":
				pass
			elif update_type == "zoom_to_point":
				ctrl.compensate_zoom_to_point(*event.data["args"], **event.data["kwargs"])

	def set_camera_state_and_draw(self, event):
		for id_other, ctrl in self._controller_group.items():
			if event.controller_id == id_other:
				continue
			ctrl.set_camera_state_and_draw()

class PynaVizController(PanZoomController):
	def __init__(
			self,
			camera: Optional[Camera] = None,
			*,
			enabled=True,
			damping: int = 4,
			auto_update: bool = True,
			register_events: Optional[Union[Viewport, Renderer]] = None,
			controller_id: Optional[int] = None
	):

		self._controller_id = controller_id

		super().__init__(camera=camera, enabled=enabled, damping=damping, auto_update=auto_update, register_events=register_events)
		self.renderer_handle_event = None
		self._draw = lambda: True
		if register_events:
			self.renderer_handle_event = self._get_event_handle(register_events)
			self._draw = lambda: self._request_draw(register_events)

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

	def _update_event(self, *args, **kwargs):
		ev = {
			"args": args,
			"kwargs": kwargs
		}
		if self.renderer_handle_event:
			self.renderer_handle_event(SyncEvent("update", controller_id=self._controller_id, data=ev))

	def _update_pan(self, delta, *, vecx, vecy):
		super()._update_pan(delta, vecx=vecx, vecy=vecy)
		self._update_event(update_type="pan", cam_state=self._get_camera_state(), delta=delta, vecx=vecx, vecy=vecy)

	def _update_zoom(self, delta):
		super()._update_zoom(delta)
		self._update_event(update_type="zoom", cam_state=self._get_camera_state(), delta=delta)

	def _update_zoom_to_point(self, delta, *, screen_pos, rect):
		super()._update_zoom_to_point(delta, screen_pos=screen_pos, rect=rect)
		self._update_event(update_type="zoom_to_point", cam_state=self._get_camera_state(), delta=delta, screen_pos=screen_pos, rect=rect)

	def compensate_pan(self, *args, **kwargs):
		cam_state = kwargs["cam_state"]
		x_pos = cam_state["position"][0]
		width = cam_state["width"]

		self_camera_state = self._get_camera_state()
		self_x_pos = self_camera_state["position"][0]
		self_width = self_camera_state["width"]
		# get the dx
		dx = (x_pos / width - self_x_pos / self_width) * self_width

		new_position = self_camera_state["position"].copy()
		new_position[0] = new_position[0] + dx

		# Update camera
		self._set_camera_state({"position": new_position})
		self._update_cameras()
		self._draw()

	def compensate_zoom_to_point(self, *args, **kwargs):
		cam_state = kwargs["cam_state"]
		self_camera_state = self._get_camera_state()

		#
		extent = 0.5 * (self_camera_state["width"] + self_camera_state["height"])
		new_extent = 0.5 * (cam_state["width"] + self_camera_state["height"])

		rot = self_camera_state["rotation"]
		fov = self_camera_state["fov"]
		distance = fov_distance_factor(fov) * extent
		v1 = la.vec_transform_quat((0, 0, -distance), rot)

		distance = fov_distance_factor(fov) * new_extent
		v2 = la.vec_transform_quat((0, 0, -distance), rot)

		new_position = self_camera_state["position"].copy()
		new_position = new_position + v1 - v2

		# Update camera
		self._set_camera_state({"position": new_position, "width": cam_state["width"]})


	def handle_event(self, event, viewport):
		super().handle_event(event, viewport)
		need_update = False
		type = event.type
		if type.startswith(("pointer_", "key_", "wheel")):
			modifiers = {m.lower() for m in event.modifiers}
			if type.startswith("key_"):
				modifiers.discard(event.key.lower())
			modifiers_prefix = "+".join(sorted(modifiers) + [""])

		if type == "before_render":
			if self._auto_update and self._actions:
				need_update = True
		elif type == "pointer_down" and viewport.is_inside(event.x, event.y):
			# Start a drag, or an action with mode push/peek/repeat
			key = modifiers_prefix + f"mouse{event.button}"
			action_tuple = self._controls.get(key)
			if action_tuple:
				need_update = True
			# Update all drag actions
			for action in self._actions.values():
				if action.mode == "drag" and not action.done:
					need_update = True
		elif type == "pointer_up":
			need_update = self._handle_button_up(f"mouse{event.button}")
		elif type == "wheel" and viewport.is_inside(event.x, event.y):
			# Wheel events. Technically there is horizontal and vertical scroll,
			# but this does not work well cross-platform, so we consider it 1D.
			key = modifiers_prefix + "wheel"
			action_tuple = self._controls.get(key)
			if action_tuple:
				need_update = True

		elif type == "key_down":
			# Start an action with mode push/peek/repeat
			key = modifiers_prefix + f"{event.key.lower()}"
			action_tuple = self._controls.get(key)
			if action_tuple:
				need_update = True

		elif type == "key_up":
			# End key presses, regardless of modifier state
			need_update = self._handle_button_up(f"{event.key.lower()}")

		if need_update:
			self.renderer_handle_event(SetStateAndDrawEvent("set_and_draw", controller_id=self._controller_id))

	def set_camera_state_and_draw(self):
		self._update_cameras()
		self._draw()