"""
This script is for generating screenshots than then will be compared to during the test
Naming of the file should equal the name of the test it corresponds to.

"""

import os

os.environ["WGPU_FORCE_OFFSCREEN"] = "1"

import pathlib

import click
import numpy as np
import pynapple as nap
from PIL import Image

import pynaviz as viz


def snapshot_tsd(path="screenshots/test_plot_tsd.png"):
    tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
    v = viz.PlotTsd(tsd1)
    v.animate()
    image_data = v.renderer.snapshot()
    image = Image.fromarray(image_data, mode="RGBA")
    image.save(path)


def snapshots_numbered_movies(path="screenshots/", path_video="test_video/"):
    for extension in ["mkv", "mp4", "avi"]:
        video_path = pathlib.Path(path_video) / f"numbered_video.{extension}"
        v = viz.PlotVideo(video_path, time=np.arange(100))

        for frame in [0, 1, 2, 99]:
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
    type=click.Choice(["tsd", "video"], case_sensitive=False),
    help="Type(s) of snapshot to generate. Can be used multiple times.",
)
@click.option(
    "--path", type=click.Path(), default="tests/screenshots/", help="Output directory."
)
@click.option(
    "--video-dir",
    type=click.Path(),
    default="tests/test_video/",
    help="Directory containing numbered videos.",
)
def main(types, path, video_dir):
    """Generate TSD or video frame snapshots."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if not types:
        click.echo("Please specify at least one --type (tsd or video).")
        return

    if "tsd" in types:
        click.echo("Generating TSD snapshot...")
        snapshot_tsd(path=path / "test_plot_tsd.png")

    if "video" in types:
        click.echo("Generating video snapshots...")
        snapshots_numbered_movies(path=path, path_video=video_dir)

    click.echo("Done.")


if __name__ == "__main__":
    main()
