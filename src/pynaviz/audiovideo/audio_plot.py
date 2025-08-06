# import pathlib
# from abc import ABC
# from typing import Any, Optional
#
# import pygfx as gfx
#
# from ..base_plot import _BasePlot
# from ..controller import SpanController
# from ..synchronization_rules import _match_pan_on_x_axis, _match_zoom_on_x_axis
# from .audio_handling import AudioHandler
#
# dict_sync_funcs = {
#     "pan": _match_pan_on_x_axis,
#     "zoom": _match_zoom_on_x_axis,
#     "zoom_to_point": _match_zoom_on_x_axis,
# }
#
#
# class PlotAudio(_BasePlot, ABC):
#     """
#     Abstract base class for time-synchronized audiovideo plots using pygfx.
#
#     Subclasses must implement the methods to provide audiovideo frames as 2D tensors.
#     """
#
#     def __init__(self, path: str | pathlib.Path, index: Optional[int] = None, parent: Optional[Any] = None) -> None:
#         """
#         Initialize the base audiovideo tensor plot.
#
#         Parameters
#         ----------
#         data : Any
#             The data source, such as a TsdTensor or a audiovideo handler.
#         index : int, optional
#             Identifier for the controller.
#         parent : Any, optional
#             Parent widget or container.
#         """
#         data = AudioHandler(path)
#         super().__init__(data, parent=parent, maintain_aspect=True)
#
#
#         texture_data = self._get_initial_texture_data()
#         self.texture = gfx.Texture(texture_data.astype("float32"), dim=2)
#
#
#         # Create a line geometry and material to render the time series
#         self.line = gfx.Line(
#             gfx.Geometry(positions=positions),
#             gfx.LineMaterial(thickness=4.0, color="#aaf"),  # light blue line
#         )
#
#         self.scene.add(self.line)
#         self.camera.show_object(self.scene)
#
#         # Create a controller for span-based interaction, syncing, and user inputs
#         self.controller = SpanController(
#             camera=self.camera,
#             renderer=self.renderer,
#             controller_id=index,
#             dict_sync_funcs=dict_sync_funcs,
#             plot_callbacks=[],
#         )
#
#         self.canvas.request_draw(
#             draw_function=lambda: self.renderer.render(self.scene, self.camera)
#         )
#
#
#     def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
#         """Placeholder for future metadata sorting method."""
#         pass
#
#     def group_by(self, metadata_name: str, spacing: Optional = None):
#         """Placeholder for future metadata grouping method."""
#         pass
#
#     @abc.abstractmethod
#     def _update_buffer(self, frame_index: int, event_type: Optional[RenderTriggerSource] = None):
#         """
#         Abstract method to update the buffer based on the frame index and event.
#
#         Parameters
#         ----------
#         frame_index : int
#             Index of the frame to load.
#         event_type : RenderTriggerSource, optional
#             Source of the event triggering the update.
#         """
#         pass
