from math import pi, tan

import matplotlib.pyplot as plt
import numpy as np
import pygfx as gfx
import pylinalg as la
from pygfx.cameras._perspective import fov_distance_factor
from wgpu.gui.auto import WgpuCanvas, run


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

colors = np.linspace(0, 1, x.shape[0]).astype(np.float32)
cmap = plt.get_cmap("Purples")
colors = cmap(colors)
colors = colors.astype(np.float32)

# colors = np.random.uniform(0, 1, (x.size, 4)).astype(np.float32)
# colors[:, 3] = 1

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
camera1 = gfx.OrthographicCamera(500, 500, maintain_aspect=False)
camera1.local.position = camera_settings1['position']
camera2 = gfx.OrthographicCamera(500, 500, maintain_aspect=False)
camera2.local.position = camera_settings2['position']

# Define scene for both canvases
scene1 = gfx.Scene()
scene2 = gfx.Scene()

line1 = gfx.Line(
    gfx.Geometry(positions=positions1, colors=colors),
    gfx.LineMaterial(thickness=3.0, color_mode="face", map=gfx.cm.viridis),
)
line2 = gfx.Line(
    gfx.Geometry(positions=positions2,colors=colors),
    gfx.LineMaterial(thickness=3.0, color_mode="face", map=gfx.cm.viridis),
)
scene1.add(line1)
scene2.add(line2)


def on_pan(event, plot, *args):

    print(event)

    plot1 = plot
    plot2 = args[0]

    camera1 = plot1["camera"]
    camera2 = plot2["camera"]

    # print("event handler 1")
    # Modify camera position based on scroll
    dx2 = (camera1.local.position[0] / camera1.width - camera2.local.position[0] / camera2.width) * camera2.width #/ camera1.width

    # update the camera setting position
    new_pos = np.copy(camera2.local.position)
    new_pos[0] = new_pos[0] + dx2
    # camera_settings2['position'][0] = camera2.local.position[0] + dx2
    # camera_settings1['position'][0] = camera1.local.position[0]

    # Update camera
    camera2.local.position = new_pos

    # Re-render both scenes
    plot2['canvas'].request_draw(lambda: plot2['renderer'].render(plot2['scene'], camera2))
    plot1['canvas'].request_draw(lambda: plot1['renderer'].render(plot1['scene'], camera1))

    print("cam1", camera1.local.position, (camera1.width, camera1.height))
    print("cam2", camera2.local.position, (camera2.width, camera2.height))

# Adding event listeners to each canvas
def on_zoom(event, plot, *args):

    plot1 = plot
    plot2 = args[0]

    camera1 = plot1["camera"]
    camera2 = plot2["camera"]

    cam_state2 = camera2.get_state()
    cam_state1 = camera1.get_state()
    extent = 0.5 * (cam_state2["width"] + cam_state2["height"])

    new_extent = 0.5 * (cam_state1["width"] + cam_state2["height"])

    rot = cam_state2["rotation"]
    fov = cam_state2["fov"]
    distance = fov_distance_factor(fov) * extent
    v1 = la.vec_transform_quat((0, 0, -distance), rot)

    distance = fov_distance_factor(fov) * new_extent
    v2 = la.vec_transform_quat((0, 0, -distance), rot)

    new_pos = np.copy(camera2.local.position)
    new_pos = new_pos + v1 - v2
    camera2.local.position = new_pos
    camera2.width = cam_state1["width"]

    plot2['canvas'].request_draw(lambda: plot2['renderer'].render(plot2['scene'], camera2))
    plot1['canvas'].request_draw(lambda: plot1['renderer'].render(plot1['scene'], camera1))
    print("cam1", camera1.local.position, (camera1.width, camera1.height))
    print("cam2", camera2.local.position, (camera2.width, camera2.height))

    on_pan(event, plot1, plot2)

# Create grouped objects

plot1 = {"camera":camera1,"renderer":renderer1,"canvas":canvas1, "scene": scene1}
plot2 = {"camera":camera2,"renderer":renderer2,"canvas":canvas2, "scene": scene2}


# Assume canvas1 and canvas2 are your canvas elements
canvas1.add_event_handler(lambda e: on_pan(e, plot1, plot2), "pointer_move")
canvas2.add_event_handler(lambda e: on_pan(e, plot2, plot1), "pointer_move")
canvas1.add_event_handler(lambda e: on_zoom(e, plot1, plot2), "wheel")
canvas2.add_event_handler(lambda e: on_zoom(e, plot2, plot1), "wheel")

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
