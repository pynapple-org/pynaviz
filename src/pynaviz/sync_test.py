import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np
from math import pi, tan
import pylinalg as la

def fov_distance_factor(fov):
    # It's important that controller and camera use the same distance calculations
    if fov > 0:
        fov_rad = fov * pi / 180
        factor = 0.5 / tan(0.5 * fov_rad)
    else:
        factor = 1.0
    return factor

# Create a renderer for each canvas
canvas1 = WgpuCanvas()
canvas2 = WgpuCanvas()
renderer1 = gfx.renderers.WgpuRenderer(canvas1)
renderer2 = gfx.renderers.WgpuRenderer(canvas2)

# data
x = np.linspace(0, 100, 10_000, dtype=np.float32)
y1 = np.sin(x)
y2 = np.sin(x)
positions1 = np.column_stack([x, y1, np.zeros_like(x)])
positions2 = np.column_stack([x, y2, np.zeros_like(x)])

# Shared camera settings
camera_settings1 = {
    'position': [0, 0, 500],
    'rotation': [0, 0, 0],
    'fov': 60
}
camera_settings2 = {
    'position': [0, 0, 500],
    'rotation': [0, 0, 0],
    'fov': 60
}

# Define two cameras with shared parameters
camera1 = gfx.OrthographicCamera(500, 400, maintain_aspect=False)
camera1.local.position = camera_settings1['position']
camera2 = gfx.OrthographicCamera(500, 400, maintain_aspect=False)
camera2.local.position = camera_settings2['position']

# Define scene for both canvases
scene1 = gfx.Scene()
scene2 = gfx.Scene()

line1 = gfx.Line(
    gfx.Geometry(positions=positions1),
    gfx.LineMaterial(thickness=2.0, color=(0.0, 0.7, 0.3, 1.0)),
)
line2 = gfx.Line(
    gfx.Geometry(positions=positions2),
    gfx.LineMaterial(thickness=2.0, color=(0.0, 0.7, 0.3, 1.0)),
)
scene1.add(line1)
scene2.add(line2)

# Add objects to scenes, etc.


def on_scroll_1(event):

    # print("event handler 1")
    # Modify camera position based on scroll
    dx2 = (camera1.local.position[0] - camera_settings1['position'][0]) * camera2.width / camera1.width

    # update the camera setting position
    camera_settings2['position'][0] = camera2.local.position[0] + dx2
    camera_settings1['position'][0] = camera1.local.position[0]

    # Update camera
    camera2.local.position = camera_settings2['position']

    # Re-render both scenes
    canvas2.request_draw(lambda: renderer2.render(scene2, camera2))
    canvas1.request_draw(lambda: renderer1.render(scene1, camera1))

    #print(camera1.local.position, (camera1.width, camera1.height), (camera2.width, camera2.height))


def on_scroll_2(event):
    # print("event handler 2")
    # Modify camera position based on scroll
    dx1 = (camera2.local.position[0] - camera_settings2['position'][0]) * camera1.width / camera2.width

    camera_settings1['position'][0] = camera1.local.position[0] + dx1
    camera_settings2['position'][0] = camera2.local.position[0]

    # Update camera
    camera1.local.position = camera_settings2['position']

    # Re-render both scenes
    canvas1.request_draw(lambda: renderer1.render(scene1, camera1))
    canvas2.request_draw(lambda: renderer2.render(scene2, camera2))

    #print(camera_settings2['position'], (camera1.width, camera1.height), (camera2.width, camera2.height))

# Adding event listeners to each canvas
def on_zoom_1(event):
    print("zoom 1")

    cam_state = camera2.get_state()
    extent = 0.5 * (cam_state["width"] + cam_state["height"])

    new_extent = 0.5 * (camera1.width + cam_state["height"])

    rot = cam_state["rotation"]
    fov = cam_state["fov"]
    distance = fov_distance_factor(fov) * extent
    v1 = la.vec_transform_quat((0, 0, -distance), rot)

    distance = fov_distance_factor(fov) * new_extent
    v2 = la.vec_transform_quat((0, 0, -distance), rot)

    camera_settings2['position'] = camera_settings2['position'] + v1 - v2
    camera2.local.position = camera_settings2['position']
    camera2.width = camera1.width

    canvas2.request_draw(lambda: renderer2.render(scene2, camera2))
    canvas1.request_draw(lambda: renderer1.render(scene1, camera1))


def on_zoom_2(event):
    print("zoom 2")

    cam_state = camera1.get_state()
    extent = 0.5 * (cam_state["width"] + cam_state["height"])

    new_extent = 0.5 * (camera2.width + cam_state["height"])

    rot = cam_state["rotation"]
    fov = cam_state["fov"]
    distance = fov_distance_factor(fov) * extent
    v1 = la.vec_transform_quat((0, 0, -distance), rot)

    distance = fov_distance_factor(fov) * new_extent
    v2 = la.vec_transform_quat((0, 0, -distance), rot)

    camera_settings1['position'] = camera_settings1['position'] + v1 - v2
    camera1.local.position = camera_settings1['position']
    camera1.width = camera2.width

    canvas1.request_draw(lambda: renderer1.render(scene1, camera1))
    canvas2.request_draw(lambda: renderer2.render(scene2, camera2))

    print(camera_settings1['position'], (camera1.width, camera1.height), (camera2.width, camera2.height))


# Assume canvas1 and canvas2 are your canvas elements
canvas1.add_event_handler(on_scroll_1, "pointer_move")
canvas2.add_event_handler(on_scroll_2, "pointer_move")
canvas1.add_event_handler(on_zoom_1, "wheel")
canvas2.add_event_handler(on_zoom_2, "wheel")

# Initial render
renderer1.render(scene1, camera1)
renderer2.render(scene2, camera2)

# canvas1.add_event_listener('click', lambda e: handle_click(e, 1))
# canvas2.add_event_listener('click', lambda e: handle_click(e, 2))

controller1 = gfx.PanZoomController(camera1, register_events=renderer1)
controller2 = gfx.PanZoomController(camera2, register_events=renderer2)

if __name__ == "__main__":
    canvas1.request_draw(lambda: renderer1.render(scene1, camera1))
    canvas2.request_draw(lambda: renderer2.render(scene2, camera2))
    run()
