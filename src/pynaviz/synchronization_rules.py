"""Available sync controller options."""

import pylinalg as la
from pygfx.cameras._perspective import fov_distance_factor

# from .controller import SyncEvent
from .events import SyncEvent


def _match_pan_on_x_axis(update_event: SyncEvent, camera_state: dict) -> dict:
    """
    Calculate an update that match panning on the x-axis.

    Parameters
    ----------
    update_event:
        The pan SyncEvent.
    camera_state:
        The camera state dictionary of the controller that needs to be sync.

    Returns
    -------
        The dictionary containing the state variable that needs to be updated.
    """

    if update_event.update_type != "pan":
        raise ValueError("Update rule/event mismatch. Update rule `_match_pan_on_x_axis` requires an event of type 'pan'.")

    other_cam_state = update_event.kwargs["cam_state"]
    x_pos = other_cam_state["position"][0]
    width = other_cam_state["width"]

    self_x_pos = camera_state["position"][0]
    self_width = camera_state["width"]

    # get the dx
    dx = (x_pos / width - self_x_pos / self_width) * self_width

    new_position = camera_state["position"].copy()
    new_position[0] = new_position[0] + dx
    return dict(position=new_position)


def _match_zoom_on_x_axis(update_event: SyncEvent, camera_state: dict) -> dict:
    """
    Calculate an update that match zoom by stretching the x-axis only,

    Parameters
    ----------
    update_event:
        The zoom/zoom_to_point SyncEvent.
    camera_state:
        The camera state dictionary of the controller that needs to be sync.

    Returns
    -------
        The dictionary containing the state variable that needs to be updated.
    """

    if update_event.update_type not in ["zoom", "zoom_to_point"]:
        raise ValueError(
            f"Update rule/event mismatch. Update rule `_match_zoom_on_x_axis` requires an event of type {['zoom', 'zoom_to_point']}."
        )

    other_cam_state = update_event.kwargs["cam_state"]

    extent = 0.5 * (camera_state["width"] + camera_state["height"])
    new_extent = 0.5 * (other_cam_state["width"] + camera_state["height"])

    rot = camera_state["rotation"]
    fov = camera_state["fov"]
    distance = fov_distance_factor(fov) * extent
    v1 = la.vec_transform_quat((0, 0, -distance), rot)

    distance = fov_distance_factor(fov) * new_extent
    v2 = la.vec_transform_quat((0, 0, -distance), rot)

    new_position = camera_state["position"].copy()
    new_position = new_position + v1 - v2

    return dict(position=new_position, width=other_cam_state["width"])
