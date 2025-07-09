"""
Generate visuals and markdown files for documentation
"""

import os

# Force offscreen rendering for headless environments (e.g., CI servers)
os.environ["WGPU_FORCE_OFFSCREEN"] = "1"

import pathlib
import sys

import click
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config



# Define base paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_SCREENSHOT_PATH = BASE_DIR / "_static/screenshots"
DEFAULT_SCREENSHOT_PATH.mkdir(parents=True, exist_ok=True)

# Markdown pathfile to write to
MARKDOWN_PATH = BASE_DIR / "user_guide"



def main():
    for conf_cls in [
        config.TsdConfig(DEFAULT_SCREENSHOT_PATH),
        config.TsdFrameConfig(DEFAULT_SCREENSHOT_PATH),
        config.TsGroupConfig(DEFAULT_SCREENSHOT_PATH),
        config.IntervalSetConfig(DEFAULT_SCREENSHOT_PATH)
    ]:
        conf_cls.run_all(fill=True)
        conf_cls.write_markdown(MARKDOWN_PATH)


if __name__ == "__main__":
    main()