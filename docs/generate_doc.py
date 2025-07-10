"""
Generate visuals and markdown files for documentation
"""

import os

from PyQt6.QtGui import QPixmap, QPainter
from PyQt6.QtWidgets import QApplication

# Force offscreen rendering for headless environments (e.g., CI servers)
os.environ["WGPU_FORCE_OFFSCREEN"] = "1"

import pathlib
import sys

import click
import numpy as np
from PIL import Image
import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config

# Define base paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_SCREENSHOT_PATH = BASE_DIR / "_static/screenshots"
DEFAULT_SCREENSHOT_PATH.mkdir(parents=True, exist_ok=True)

# Markdown pathfile to write to
MARKDOWN_PATH = BASE_DIR / "user_guide"

# def main_qt():
#     app = QApplication.instance() or QApplication(sys.argv)
#
#     for conf_cls in [
#         config.TsdConfig(DEFAULT_SCREENSHOT_PATH),
#         config.TsdFrameConfig(DEFAULT_SCREENSHOT_PATH),
#         config.TsGroupConfig(DEFAULT_SCREENSHOT_PATH),
#         config.IntervalSetConfig(DEFAULT_SCREENSHOT_PATH),
#     ]:
#         data = conf_cls.get_data()
#         object_name = data.__class__.__name__
#         widget_cls = getattr(viz, f"{object_name}Widget")
#         widget = widget_cls(data)
#
#         widget.resize(640, 480)  # Set a reasonable size for offscreen rendering
#         app.processEvents()      # Ensure layout is updated
#
#         # Offscreen rendering to pixmap
#         pixmap = QPixmap(widget.size())
#         pixmap.fill()  # optional: fill with white background
#         painter = QPainter(pixmap)
#         widget.render(painter)
#         painter.end()
#
#         # Save the rendered widget to file
#         filename = f"test_{object_name}Widget.png"
#         output_path = conf_cls.path / filename
#         pixmap.save(str(output_path))

def main():
    for conf_cls in [
        config.TsdConfig(DEFAULT_SCREENSHOT_PATH),
        config.TsdFrameConfig(DEFAULT_SCREENSHOT_PATH),
        config.TsGroupConfig(DEFAULT_SCREENSHOT_PATH),
        config.IntervalSetConfig(DEFAULT_SCREENSHOT_PATH)
    ]:
        conf_cls.run_all(fill=True)
        conf_cls.write_simple_visuals(MARKDOWN_PATH)


if __name__ == "__main__":
    main()
    # main_qt()