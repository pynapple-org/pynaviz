import inspect
import os
from typing import TYPE_CHECKING, Callable, Union

from pygfx import Renderer, Viewport

if TYPE_CHECKING:
    from .base_plot import _BasePlot

from enum import Enum

from pylinalg import vec_transform, vec_unproject

GRADED_COLOR_LIST = [
    "navy",
    "blue",
    "royalblue",
    "cornflowerblue",
    "skyblue",
    "lightblue",
    "aquamarine",
    "mediumseagreen",
    "limegreen",
    "yellowgreen",
    "gold",
    "orange",
    "darkorange",
    "tomato",
    "orangered",
    "red",
    "crimson",
    "deeppink",
    "magenta",
]


class RenderTriggerSource(Enum):
    """Enumeration of the renderer draw triggering source."""

    UNKNOWN = 0
    INITIALIZATION = 1
    ZOOM_TO_POINT = 2
    SYNC_EVENT_RECEIVED = 3
    LOCAL_KEY = 4
    SET_FRAME = 5

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


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


def get_plot_attribute(
    plot: "_BasePlot", attr_name, filter_graphic: dict[bool] = None
) -> dict | None:
    """Auxiliary safe function for debugging."""
    graphic = getattr(plot, "graphic", None)
    if graphic is None:
        print(f"{plot} doesn't have a graphic.")
        return None
    if isinstance(graphic, dict):
        filter_graphic = filter_graphic or {c: True for c in graphic}
        dict_attr: dict = {
            c: getattr(graphic[c], attr_name)
            for c in graphic
            if hasattr(graphic[c], attr_name) and filter_graphic[c]
        }
        return dict_attr
    else:
        if hasattr(graphic, attr_name):
            return getattr(graphic, attr_name)
        else:
            print(f"{graphic} doesn't have attribute {attr_name}.")
            return None


def trim_kwargs(func, kwargs):
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}



def map_screen_to_world(camera, pos, viewport_size):
    # first convert position to NDC
    x = pos[0] / viewport_size[0] * 2 - 1
    y = -(pos[1] / viewport_size[1] * 2 - 1)
    pos_ndc = (x, y, 0)
    pos_ndc += vec_transform(camera.world.position, camera.camera_matrix)
    # unproject to world space
    pos_world = vec_unproject(pos_ndc[:2], camera.camera_matrix)
    return pos_world


def get_plot_min_max(plot):
    """
    Get xmin,xmax, ymin, ymax in world coordinates.

    Parameters
    ----------
    plot:
        The plot object.

    Returns
    -------
    :
        The plot xmin, xmax, ymin, ymax in world coordinates.

    """
    xmin, ymin = 0, plot.renderer.logical_size[1]
    xmax, ymax = plot.renderer.logical_size[0], 0

    # Given the camera position and the range of screen space, convert to world space.
    # Get the bottom corner and top corner
    world_xmin, world_ymin, _ = map_screen_to_world(
        plot.camera, pos=(xmin, ymin), viewport_size=plot.renderer.logical_size
    )
    world_xmax, world_ymax, _ = map_screen_to_world(
        plot.camera, pos=(xmax, ymax), viewport_size=plot.renderer.logical_size
    )
    return world_xmin, world_xmax, world_ymin, world_ymax


def check_processes():
    """Simple function to check all Python processes.

    Common usage: while using the debugger, call to check which python processes are
    active after each line of code.
    Note that you may need to wait a few seconds when a new process is started.

    Notes
    -----
    Needs psutils
    """
    import psutil
    current_pid = os.getpid()
    print(f"\n{'=' * 80}")
    print(f"Current process PID: {current_pid}")
    print(f"{'=' * 80}")

    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'ppid', 'status']):
        try:
            if 'python' in proc.info['name'].lower():
                python_procs.append(proc.info)
        except Exception:
            print("Failed to get process info.")

    print(f"Found {len(python_procs)} Python processes:\n")

    for p in python_procs:
        marker = " <- YOU" if p['pid'] == current_pid else ""
        parent_marker = " <- YOUR CHILD" if p['ppid'] == current_pid else ""
        print(
            f"PID: {p['pid']:6d} | Parent: {p['ppid']:6d} | Status: {p['status']:10s} | {p['name']}{marker}{parent_marker}")

    print(f"{'=' * 80}\n")
    return len(python_procs)
