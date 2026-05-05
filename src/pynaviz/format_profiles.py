"""
Format profiles for automatic sort/group/color on first plot.

FORMAT_PROFILES maps format name → widget class name → FormatProfile.
This lets each format define different defaults per data type (e.g. TsdFrame
vs TsGroup), while remaining a no-op for types not listed.

To add defaults for a new format or data type, add an entry here — no other
file needs to change.
"""

from dataclasses import dataclass


@dataclass
class FormatProfile:
    """Default sort/group/color actions applied when a format is first plotted."""

    group_by: str | None = None
    sort_by: str | None = None
    color_by: str | None = None
    sort_mode: str = "descending"  # or "ascending"
    cmap_name: str | None = None  # e.g. "tab10", "viridis", etc.


FORMAT_PROFILES: dict[str, dict[str, FormatProfile]] = {
    "NeuroScopeIO": {
        "TsdFrameWidget": FormatProfile(
            group_by="group",
            sort_by="anatomy",
            color_by="group",
            cmap_name="tab10",
        ),
    },
}
