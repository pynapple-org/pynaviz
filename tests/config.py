"""
Config classes to generate data and views
"""


import re
from pathlib import Path

import numpy as np
import pynapple as nap
from PIL import Image

import uviz as viz


class TsdFrameConfig:
    """
    Configuration class for testing PlotTsdFrame visualizations.
    Generates example data, applies visualization methods, and saves snapshots.
    """

    metadata = {
        "group": [0, 0, 1, 0, 1],
        "channel": [1, 3, 0, 2, 4],
        "random": np.random.randn(5),
    }

    parameters = [
        (None, {}),
        ("group_by", {"metadata_name": "group"}),
        ("sort_by", {"metadata_name": "channel"}),
        ("color_by", {"metadata_name": "random"}),
        (["group_by", "sort_by"], [{"metadata_name": "group"},{"metadata_name": "channel"}]),
        (["group_by", "sort_by", "color_by"], [{"metadata_name": "group"}, {"metadata_name": "channel"}, {"metadata_name": "random"}]),
        ("plot_x_vs_y", {"x_label": 0, "y_label": 1}),
    ]

    def __init__(self, path):
        """
        Parameters
        ----------
        path : str or pathlib.Path
            Directory where the snapshots will be saved.
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def tsdframe():
        """
        Create a synthetic TsdFrame with cosine signals and metadata.
        """
        t = np.arange(0, 10, 0.01)
        offsets = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        d = np.cos(t[None, :]*4*np.pi + offsets[:, None])
        return nap.TsdFrame(t=t, d=d.T, metadata=TsdFrameConfig.metadata)

    @staticmethod
    def _build_filename(name, kwargs):
        """
        Construct a safe filename based on function name and kwargs.
        """
        raw = "test_plot_tsdframe"
        if name is not None and len(name):
            if isinstance(name, (list, tuple)):
                for n, kw in zip(name, kwargs):
                    raw += "_" + n + "_" + "_".join(f"{k}={v}" for k, v in kw.items())
            else:
                raw += "_" + name + "_" + "_".join(f"{k}={v}" for k, v in kwargs.items())
        safe = re.sub(r"[^\w\-_=\.]", "_", raw)
        return safe + ".png"

    def _save_snapshot(self, viewer, name, kwargs):
        """
        Render and save a snapshot of the viewer.
        """
        viewer.animate()
        image = Image.fromarray(viewer.renderer.snapshot())
        filename = self._build_filename(name, kwargs)
        image.save(self.path / filename)

    def run_all(self):
        """
        Run all test parameters on the synthetic TsdFrame and save results.
        """
        tsdframe = self.tsdframe()
        for name, kwargs in self.parameters:
            viewer = viz.PlotTsdFrame(tsdframe)
            if name is not None:
                if isinstance(name, (list, tuple)):
                    for n, k in zip(name, kwargs):
                        getattr(viewer, n)(**k)
                else:
                    getattr(viewer, name)(**kwargs)

                self._save_snapshot(viewer, name, kwargs)




def tsd():
    return nap.Tsd(t=np.arange(0, 10, 0.1),
                   d=np.sin(np.arange(0, 10, 0.1))
                   )

def intervalset():
    return nap.IntervalSet(
        [0, 0.2, 0.4, 0.6, 0.8],
        [0.19, 0.39, 0.59, 0.79, 0.99],
        metadata={
            "label": ["a", "b", "c", "d", "e"],
            "choice": [1, 0, 1, 1, 0],
            "reward": [0, 0, 1, 0, 1],
        },
    )
