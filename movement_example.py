"""
Movement → Pynapple minimal example
Based on: https://movement.neuroinformatics.dev/latest/examples/scale.html
"""

import warnings
from movement import sample_data
from movement.filtering import filter_by_confidence, interpolate_over_time, median_filter
from movement.transforms import scale
import pynapple as nap
from pynaviz import scope
from pynaviz.skeleton_geometry import resolve_keypoint_labels

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load and pre-process
# ---------------------------------------------------------------------------

ds = sample_data.fetch_dataset("DLC_single-mouse_DBTravelator_2D.predictions.h5")

kp_dim  = "keypoint"   if "keypoint"   in ds.coords else "keypoints"
ind_dim = "individual" if "individual" in ds.coords else "individuals"

landmark_keypoints = [
    "Door", "StartPlatL", "StartPlatR", "StepL", "StepR", "TransitionL", "TransitionR",
]

ds_mouse = ds.sel(
    {
        kp_dim:  ~ds[kp_dim].isin(landmark_keypoints),
        ind_dim: ds[ind_dim].values[0],
    }
)

ds_mouse["position"] = filter_by_confidence(ds_mouse.position, ds_mouse.confidence, threshold=0.9)
ds_mouse["position"] = interpolate_over_time(ds_mouse.position, max_gap=40)
ds_mouse["position"] = median_filter(ds_mouse.position, window=6, min_periods=2)

# Scale pixels → cm (1 cm grid ≈ 29 px) and flip y-axis
# ds_mouse["position"] = scale(ds_mouse["position"], factor=1.0 / 29.0, space_unit="cm")
y = ds_mouse["position"].sel(space="y")
ds_mouse["position"].loc[dict(space="y")] = y.max() - y

# ---------------------------------------------------------------------------
# Convert to pynapple
# ---------------------------------------------------------------------------

data = nap.from_movement(ds_mouse)
print(data)

position   = data["position"]    # TsdFrame — columns: Nose_x, Nose_y, EarL_x, ...
confidence = data["confidence"]  # TsdFrame — columns: Nose, EarL, ...

print(position)

skeleton = [
    ("Nose", "Back1"),
    ("EarL", "Back1"),
    ("EarR", "Back1"),
    ("Back1", "Back2"),
    ("Back2", "Back3"),
    ("Back3", "Back4"),
    ("Back4", "Back5"),
    ("Back5", "Back6"),
    ("Back6", "Back7"),
    ("Back7", "Back8"),
    ("Back8", "Back9"),
    ("Back9", "Back10"),
    ("Back10", "Back11"),
    ("Back11", "Back12"),
    ("Back12", "Tail1"),
    ("Tail1", "Tail2"),
    ("Tail2", "Tail3"),
    ("Tail3", "Tail4"),
    ("Tail4", "Tail5"),
    ("Tail5", "Tail6"),
    ("Tail6", "Tail7"),
    ("Tail7", "Tail8"),
    ("Tail8", "Tail9"),
    ("Tail9", "Tail10"),
    ("Tail10", "Tail11"),
    ("Tail11", "Tail12"),
    ("Back3", "ForepawKneeL"),
    ("Back3", "ForepawKneeR"),
    ("Back9", "HindpawKneeL"),
    ("Back9", "HindpawKneeR"),
    ("ForepawKneeL", "ForepawAnkleL"),
    ("ForepawKneeR", "ForepawAnkleR"),
    ("HindpawKneeL", "HindpawAnkleL"),
    ("HindpawKneeR", "HindpawAnkleR"),
    ("ForepawAnkleL", "ForepawKnuckleL"),
    ("ForepawAnkleR", "ForepawKnuckleR"),
    ("HindpawAnkleL", "HindpawKnuckleL"),
    ("HindpawAnkleR", "HindpawKnuckleR"),
    ("ForepawKnuckleL", "ForepawToeL"),
    ("ForepawKnuckleR", "ForepawToeR"),
    ("HindpawKnuckleL", "HindpawToeL"),
    ("HindpawKnuckleR", "HindpawToeR"),
]


def set_skeleton_metadata(tsdframe, edges):
    """Attach a (child, parent) skeleton edge list to a TsdFrame as "parent" metadata.

    One "parent" value is stored per column, shared by a keypoint's `_x`/`_y`
    pair. Keypoints absent from `edges` (i.e. the root) get parent `None`.
    """
    labels = resolve_keypoint_labels(tsdframe.columns)
    parent_of = dict(edges)
    parents = [parent_of.get(label) for label in labels for _ in range(2)]
    tsdframe.set_info(parent=parents)
    return tsdframe


set_skeleton_metadata(position, skeleton)

scope({"pos": position, "conf": confidence, "video": "single-mouse_DBTravelator_video.avi"}, layout_path = "layout_2026-07-14_17-58.json")