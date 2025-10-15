"""
Config classes to generate data and views
"""
import hashlib
import pathlib
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pynapple as nap
from PIL import Image

import pynaviz as viz
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def fill_background(img_path: str, background_color=(255, 255, 255)):
    img = Image.open(img_path).convert("RGBA")
    background = Image.new("RGBA", img.size, background_color + (255,))
    composed = Image.alpha_composite(background, img)
    return composed.convert("RGB")


class BaseConfig:
    """
    Configuration class for testing visualizations.
    Generates example data, applies visualization methods, and saves snapshots.
    """
    parameters = None
    def __init__(self, path):
        """
        Parameters
        ----------
        path : str or pathlib.Path
            Directory where the snapshots will be saved.
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _build_filename(cls, name, kwargs, max_length=120):
        """
        Construct a safe filename based on function name and kwargs.
        """
        basename = cls.__name__.removesuffix("Config").lower()
        raw = f"test_plot_{basename}"
        if name is not None and len(name):
            if isinstance(name, (list, tuple)):
                for n, kw in zip(name, kwargs):
                    raw += "_" + n + "_" + "_".join(f"{k}={type(v)}" for k, v in kw.items())
            else:
                raw += "_" + name + "_" + "_".join(f"{k}={type(v)}" for k, v in kwargs.items())
        safe = re.sub(r"[^\w\-_=\.]", "_", raw)

        # If too long hash the tail for uniqueness
        if len(safe) > max_length:
            digest = hashlib.md5(safe.encode()).hexdigest()[:8]
            safe = safe[: max_length - 9] + "_" + digest

        return safe + ".png"

    def _save_snapshot(self, viewer, name, kwargs, fill=False):
        """
        Render and save a snapshot of the viewer.
        """
        viewer.animate()
        image = Image.fromarray(viewer.renderer.snapshot())
        if fill:
            background = Image.new("RGBA", image.size, (0,0,0,255))
            image = Image.alpha_composite(background, image)
        filename = self._build_filename(name, kwargs)

        image.save(self.path / filename)

    def run_all(self, fill=False):
        """
        Run all
        """
        data = self.get_data()
        object_name = data.__class__.__name__
        for name, kwargs in self.parameters:
            if object_name == "VideoHandler":
                viewer = viz.PlotVideo(data)
            else:
                viewer = getattr(viz, f"Plot{object_name}")(data)

            if name is not None:
                if isinstance(name, (list, tuple)):
                    for n, k in zip(name, kwargs):
                        getattr(viewer, n)(**k)
                else:
                    getattr(viewer, name)(**kwargs)

            self._save_snapshot(viewer, name, kwargs, fill)
            viewer.close()

    @staticmethod
    def get_data():
        pass

    def write_simple_visuals(self, path):
        """
        """
        obj_repr = self.get_data().__repr__()
        obj_name = self.get_data().__class__.__name__
        header = textwrap.dedent(f"""\
# {obj_name}

```python
import pynapple as nap
import pynaviz as viz

print({obj_name.lower()})
```
```
{obj_repr}
```
        """)

        md_path = path / f"{obj_name.lower()}.md"
        md_path.write_text(header.strip(), encoding="utf-8")

        with md_path.open("a", encoding="utf-8") as f:
            for name, kwargs in self.parameters:
                if name is None or isinstance(name, str):  # Doing just the single action
                    filename = self._build_filename(name, kwargs)
                    cmd = ""
                    if isinstance(name, str):
                        cmd += "v." + name
                    if len(kwargs):
                        cmd += "("
                        str_kwargs = []
                        for k, v in kwargs.items():
                            if isinstance(v, str):
                                str_kwargs.append(k + "='" + v + "'")
                            elif not isinstance(v, (int, float, bool)):
                                str_kwargs.append(k + "=" + str(v.__class__.__name__))
                            else:
                                str_kwargs.append(k + "=" + str(v))

                        cmd += ", ".join(str_kwargs) + ")"
                    block = f"""

---
## Plot{obj_name}{" - " + name if name else ""}                    

```python
v = viz.Plot{obj_name}({obj_name.lower()})
{cmd}
v.show()
```
![{name}](/_static/screenshots/{filename})

                    """
                    f.write(block)



class TsdFrameConfig(BaseConfig):
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
        ("plot_x_vs_y", {"x_col": 0, "y_col": 1}),
        (["group_by", "_reset"], [{"metadata_name": "group"}, {"event": SimpleNamespace(type="key_down",key="r")}]),
        (["plot_x_vs_y", "_reset"], [{"x_col": 0, "y_col": 1}, {"event": SimpleNamespace(type="key_down", key="r")}]),
        (["sort_by", "_rescale"], [{"metadata_name": "channel"}, {"event": SimpleNamespace(type="key_down", key="i")}]),
        (["sort_by", "_rescale"], [{"metadata_name": "channel"}, {"event": SimpleNamespace(type="key_down", key="d")}]),
        ("add_interval_sets", {"epochs": nap.IntervalSet([0.25, 0.4, 0.7, 0.8])}),
    ]
    @staticmethod
    def get_data():
        """
        Create a synthetic TsdFrame with cosine signals and metadata.
        """
        t = np.arange(0, 10, 0.01)
        offsets = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        d = np.cos(t[None, :]*4*np.pi + offsets[:, None])
        return nap.TsdFrame(t=t, d=d.T, metadata=TsdFrameConfig.metadata)


class IntervalSetConfig(BaseConfig):
    metadata = {
        "label": ["a", "b", "c", "d", "e"],
        "choice": [1, 0, 1, 1, 0],
        "reward": [0, 0, 1, 0, 1],
    }
    parameters = [
        (None, {}),
        ("group_by", {"metadata_name": "choice"}),
        ("sort_by", {"metadata_name": "label"}),
        ("color_by", {"metadata_name": "reward"}),
        (["group_by", "sort_by"], [{"metadata_name": "choice"}, {"metadata_name": "label"}]),
        (["group_by", "sort_by", "color_by"],
         [{"metadata_name": "label"}, {"metadata_name": "choice"}, {"metadata_name": "reward"}]),
    ]

    @staticmethod
    def get_data():
        return nap.IntervalSet(
            [0, 0.2, 0.4, 0.6, 0.8],
            [0.19, 0.39, 0.59, 0.79, 0.99],
            metadata=IntervalSetConfig.metadata
        )


class TsdConfig(BaseConfig):
    parameters = [
        (None, {}),
        ("add_interval_sets", {"epochs": nap.IntervalSet([0.25, 0.4, 0.7, 0.8])}),
    ]

    @staticmethod
    def get_data():
        t = np.arange(0, 2, 0.01)
        return nap.Tsd(t=t,d=np.cos(t*6*np.pi))


class TsGroupConfig(BaseConfig):
    metadata = {
        "group": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        "channel": [1, 3, 0, 2, 4, 5, 6, 7, 8, 9],
        "random": np.random.randn(10),
    }
    parameters = [
        (None, {}),
        ("group_by", {"metadata_name": "group"}),
        ("sort_by", {"metadata_name": "rate"}),
        ("color_by", {"metadata_name": "group"}),
        (["group_by", "sort_by"], [{"metadata_name": "group"},{"metadata_name": "rate"}]),
        (["group_by", "sort_by", "color_by"], [{"metadata_name": "group"}, {"metadata_name": "rate"}, {"metadata_name": "random"}]),
        ("add_interval_sets", {"epochs": nap.IntervalSet([0.25, 0.4, 0.7, 0.8])}),
    ]
    @staticmethod
    def get_data():
        return nap.TsGroup({
            i: nap.Ts(
                t=np.sort(np.random.uniform(0, 100, 100 * (i + 1)))
            ) for i in range(10)},
            metadata=TsGroupConfig.metadata)


class TsConfig(BaseConfig):
    parameters = [
        (None, {}),
        ("add_interval_sets", {"epochs": nap.IntervalSet([0.25, 0.4, 0.7, 0.8])}),
    ]

    @staticmethod
    def get_data():
        t = np.array([0.1, 0.3, 0.5, 0.55, 0.9])
        return nap.Ts(t=t)


class TsdTensorConfig(BaseConfig):
    parameters = [
        (None, {}),
        ("superpose_points", {"tsdframe":
            nap.TsdFrame(
                t=np.arange(3),
                d=np.random.uniform(0, 50, size=(3, 8)),
                )
            }
        ),
    ]

    @staticmethod
    def get_data():
        t = np.arange(3)
        n = 50
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, y)
        d = np.sin(X ** 2 + Y ** 2) + np.cos(3 * X) * np.sin(3 * Y)
        d = np.repeat(d[None,:], 3, 0)
        return nap.TsdTensor(t=t, d=d)

class VideoHandlerConfig(BaseConfig):
    parameters = [
        (None, {}),
    ]

    @staticmethod
    def get_data():

        base_dir = pathlib.Path(__file__).parent.resolve()
        save_path = base_dir / "docs/_static/855401-uhd_3840_2160_25fps.mp4"

        return viz.VideoHandler(save_path)