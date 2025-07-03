"""
This script generates screenshots that are compared during tests.
The name of each file should match the corresponding test.

Run with:
    python tests/generate_screenshots.py --type tsd --type video
"""

import os

# Force offscreen rendering for headless environments (e.g., CI servers)
os.environ["WGPU_FORCE_OFFSCREEN"] = "1"

import pathlib
import click
import numpy as np
import pynapple as nap
from PIL import Image
import conftest as conf
import uviz as viz

# Define base paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_SCREENSHOT_PATH = BASE_DIR / "screenshots"
DEFAULT_VIDEO_DIR = BASE_DIR / "test_video"

# ---------- Snapshot classes ----------
class SnapshotsTsdFrame:
    """
    Generate screenshots for different calls to PlotTsdFrame.
    """

    def __init__(self, path):
        self.tsdframe = conf.tsdframe()
        self.path = path

    def _save_snapshot(self, viewer, name):
        viewer.animate()
        image = Image.fromarray(viewer.renderer.snapshot())
        image.save(self.path / f"{name}.png")

    def run_all(self):
        actions = {
            "test_plot_tsdframe": lambda v: None,
            "test_plot_tsdframe_sort_by": lambda v: v.sort_by("channel"),
            "test_plot_tsdframe_group_by": lambda v: v.group_by("group"),
            # "test_plot_tsdframe_color_by": lambda v: v.color_by("group"),
            "test_plot_tsdframe_all": lambda v: (v.group_by("group"), v.sort_by("channel")),
            # "test_plot_tsdframe_reset": lambda v: (v.group_by("random"), v.sort_by("channel"), v._reset()),
            "test_plot_tsdframe_plot_x_vs_y": lambda v: v.plot_x_vs_y(0, 1),
        }

        for name, action in actions.items():
            v = viz.PlotTsdFrame(self.tsdframe)
            action(v)
            self._save_snapshot(v, name)


# ---------- Snapshot functions ----------
def snapshot_tsd(path=DEFAULT_SCREENSHOT_PATH / "test_plot_tsd.png"):
    """
    Generate and save a snapshot of a Tsd plot.
    """
    tsd1 = conf.tsd()
    v = viz.PlotTsd(tsd1)
    v.animate()
    image_data = v.renderer.snapshot()
    image = Image.fromarray(image_data)#, mode="RGBA")
    image.save(path)

def snapshot_intervalset(path=DEFAULT_SCREENSHOT_PATH / "test_plot_intervalset.png"):
    """
    Generate and save a snapshot of an IntervalSet plot.
    """
    ep = conf.intervalset()
    v = viz.PlotIntervalSet(ep)
    v.animate()
    image_data = v.renderer.snapshot()
    image = Image.fromarray(image_data)#, mode="RGBA")
    image.save(path)

def snapshot_tsdframe(path=DEFAULT_SCREENSHOT_PATH):
    """
    """
    print(path)
    snapshots = SnapshotsTsdFrame(path)
    snapshots.run_all()

def snapshots_numbered_movies(path=DEFAULT_SCREENSHOT_PATH, path_video=DEFAULT_VIDEO_DIR, frames=None):
    """
    Generate and save snapshots of specific frames from numbered videos
    (supports mkv, mp4, avi formats).
    """
    if frames is None:
        # Default frames to snapshot
        frames = [0, 1, 2, 3, 4, 10, 12, 14, 16, 18, 25, 50, 75, 95, 96, 97, 98, 99]

    path = pathlib.Path(path) / "video/"
    path.mkdir(parents=True, exist_ok=True)

    for extension in ["mkv", "mp4", "avi"]:
        video_path = pathlib.Path(path_video) / f"numbered_video.{extension}"
        v = viz.PlotVideo(video_path, t=np.arange(100))

        for frame in frames:
            # Build output path for each frame image
            path_frame = pathlib.Path(path) / f"numbered_video_{extension}_frame_{frame}.png"
            v.set_frame(frame)
            v.renderer.render(v.scene, v.camera)
            image_data = v.renderer.snapshot()
            image = Image.fromarray(image_data)#, mode="RGBA")
            image.save(path_frame)

# ---------- CLI entry point using Click ----------

@click.command()
@click.option(
    "--type",
    "types",
    multiple=True,
    type=click.Choice(["tsd", "tsdframe", "video", "all"], case_sensitive=False),
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
    """
    Main function that handles snapshot generation based on CLI options.
    """
    # Convert strings to Path objects
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if not types:
        click.echo("Please specify at least one --type (tsd or video).")
        return

    # Parse frame indices if provided
    frame_list = None
    if frames:
        frame_list = [int(f.strip()) for f in frames.split(",") if f.strip().isdigit()]

    # Generate TSD snapshot
    if "tsd" in types or "all" in types:
        click.echo("Generating Tsd snapshot...")
        snapshot_tsd(path=path / "test_plot_tsd.png")

    # Generate ISet snapshot
    if "intervalset" in types or "all" in types:
        click.echo("Generating Intervalset snapshot...")
        snapshot_intervalset(path=path / "test_plot_intervalset.png")

    if "tsdframe" in types or "all" in types:
        click.echo("Generating TsdFrame snapshot...")
        snapshot_tsdframe(path=path)

    # Generate video frame snapshots
    if "video" in types or "all" in types:
        click.echo("Generating video snapshots...")
        snapshots_numbered_movies(path=path, path_video=video_dir, frames=frame_list)

    click.echo("Done.")

# ---------- Script entry point ----------

if __name__ == "__main__":
    main()
