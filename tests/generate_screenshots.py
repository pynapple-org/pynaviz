"""
This script generates screenshots that are compared during tests.
The name of each file should match the corresponding test.

Run with:
    python tests/generate_screenshots.py --type tsd --type video
"""

import os

# Force offscreen rendering for headless environments
os.environ["WGPU_FORCE_OFFSCREEN"] = "1"

import pathlib

import click
import numpy as np
import pynapple as nap
from PIL import Image

import pynaviz as viz

# Base directory of the project (assuming tools/ is at the root)
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_SCREENSHOT_PATH = BASE_DIR / "screenshots"
DEFAULT_VIDEO_DIR = BASE_DIR / "test_video"


def snapshot_tsd(path=DEFAULT_SCREENSHOT_PATH / "test_plot_tsd.png"):
    tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
    v = viz.PlotTsd(tsd1)
    v.animate()
    image_data = v.renderer.snapshot()
    image = Image.fromarray(image_data, mode="RGBA")
    image.save(path)

# v = viz.PlotTsd(tsd1)
# v.animate()
# image_data = v.renderer.snapshot()
# image = Image.fromarray(image_data, mode="RGBA")
# image.save(os.path.expanduser("~/pynaviz/tests/screenshots/test_plot_tsd.png"))


# ep = nap.IntervalSet(
#     [0, 0.2, 0.4, 0.6, 0.8],
#     [0.19, 0.39, 0.59, 0.79, 0.99],
#     metadata={
#         "label": ["a", "b", "c", "d", "e"],
#         "choice": [1, 0, 1, 1, 0],
#         "reward": [0, 0, 1, 0, 1],
#     },
# )
# v = viz.PlotIntervalSet(ep)
# v.animate()
# image_data = v.renderer.snapshot()
# image = Image.fromarray(image_data, mode="RGBA")
# image.save(
#     os.path.expanduser("~/GitHub/pynaviz/tests/screenshots/test_plot_intervalset.png")
# >>>>>>> dev    
    

def snapshots_numbered_movies(path=DEFAULT_SCREENSHOT_PATH, path_video=DEFAULT_VIDEO_DIR, frames=None):
    if frames is None:
        frames = [0, 1, 2, 3, 4,10, 12, 14, 16, 18, 25, 50, 75, 95, 96, 97, 98, 99]

    for extension in ["mkv", "mp4", "avi"]:
        video_path = pathlib.Path(path_video) / f"numbered_video.{extension}"
        v = viz.PlotVideo(video_path, t=np.arange(100))

        for frame in frames:
            path_frame = (
                pathlib.Path(path) / f"numbered_video_{extension}_frame_{frame}.png"
            )
            v.set_frame(frame)
            v.renderer.render(v.scene, v.camera)
            image_data = v.renderer.snapshot()
            image = Image.fromarray(image_data, mode="RGBA")
            image.save(path_frame)


@click.command()
@click.option(
    "--type",
    "types",
    multiple=True,
    type=click.Choice(["tsd", "video", "all"], case_sensitive=False),
    help="Type(s) of snapshot to generate. Can be used multiple times.",
)
@click.option(
    "--path", type=click.Path(), default=str(DEFAULT_SCREENSHOT_PATH),
    help="Output directory for snapshots.",

)
@click.option(
    "--frames",
    type=str,
    default=None,
    help="Comma-separated list of frame indices to render (e.g. 0,1,2,99). Only applies to video.",
)
@click.option(
    "--video-dir",
    type=click.Path(),
    default=str(DEFAULT_VIDEO_DIR),
    help="Directory containing numbered videos.",
)
def main(types, path, video_dir, frames):
    """Generate TSD or video frame snapshots."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if not types:
        click.echo("Please specify at least one --type (tsd or video).")
        return

    frame_list = None
    if frames:
        frame_list = [int(f.strip()) for f in frames.split(",") if f.strip().isdigit()]

    if "tsd" in types or "all" in types:
        click.echo("Generating Tsd snapshot...")
        snapshot_tsd(path=path / "test_plot_tsd.png")

    if "video" in types or "all" in types:
        click.echo("Generating video snapshots...")
        snapshots_numbered_movies(path=path, path_video=video_dir, frames=frame_list)

    click.echo("Done.")


if __name__ == "__main__":
    main()
