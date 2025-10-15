"""
Generate visuals and markdown files for documentation
"""

import os

# Force offscreen rendering for headless environments (e.g., CI servers)
os.environ["WGPU_FORCE_OFFSCREEN"] = "1"

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config

# Define base paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_SCREENSHOT_PATH = BASE_DIR / "_static/screenshots"
DEFAULT_SCREENSHOT_PATH.mkdir(parents=True, exist_ok=True)

# Markdown pathfile to write to
MARKDOWN_PATH = BASE_DIR / "user_guide"

def combine_gifs(gif_paths, output_path, duration=500):
    from PIL import Image

    frames = []

    for gif_path in gif_paths:
        with Image.open(gif_path) as img:
            try:
                while True:
                    frames.append(img.copy())
                    img.seek(img.tell() + 1)
            except EOFError:
                pass  # End of sequence

    if frames:
        gif0 = Image.open(gif_paths[0])
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=gif0.info.get('duration', 100),
            loop=0,
        )

def main():
    for conf_cls in [
        config.TsdConfig(DEFAULT_SCREENSHOT_PATH),
        config.TsdFrameConfig(DEFAULT_SCREENSHOT_PATH),
        config.TsGroupConfig(DEFAULT_SCREENSHOT_PATH),
        config.IntervalSetConfig(DEFAULT_SCREENSHOT_PATH),
        config.TsConfig(DEFAULT_SCREENSHOT_PATH),
        config.TsdTensorConfig(DEFAULT_SCREENSHOT_PATH),
        config.VideoHandlerConfig(DEFAULT_SCREENSHOT_PATH),
    ]:
        conf_cls.run_all(fill=True)
        conf_cls.write_simple_visuals(MARKDOWN_PATH)

    # # Combining all gifs into one file
    # gifs_path = BASE_DIR / "examples/"
    # combined_gif_path = BASE_DIR / "_static/combined_widgets.gif"
    # combine_gifs(
    #     [p for p in gifs_path.glob("*.gif") if "combined" not in p.name],
    #     combined_gif_path,
    #     duration=800,
    # )
    # print(f"Combined GIF saved to {combined_gif_path}")


if __name__ == "__main__":
    main()


