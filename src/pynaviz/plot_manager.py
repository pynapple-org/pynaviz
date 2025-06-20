"""
Plot manager for TsGroup, TsdFrame, and IntervalSet visualizations.
The manager gives context for which action has been applied to the visual.
"""

import numpy as np
from pynapple.core.metadata_class import _Metadata


class _PlotManager:
    """
    Manages the plotting state for visual elements like TsGroup, TsdFrame, and IntervalSet.

    Tracks the following per-element metadata:
        - `groups`: group labels from group_by action.
        - `order`: display order from sort_by action.
        - `visible`: visibility status of each element.
        - `offset`: vertical offset per element.
        - `scale`: scale multiplier per element.
    """

    def __init__(self, index: list | np.ndarray):
        """
        Initializes the plot manager with default metadata values for a given index.

        Parameters
        ----------
        index : list or np.ndarray
            Index of elements—e.g., TsGroup keys, TsdFrame columns, or IntervalSet rows.
        """
        self.index = index
        self.data = _Metadata(
            index=index,
            data={
                "groups": np.zeros(len(index), dtype=int),
                "order": np.zeros(len(index), dtype=int),
                "visible": np.ones(len(index), dtype=bool),
                "offset": np.zeros(len(index)),
                "scale": np.ones(len(index)),
            },
        )
        # To keep track of past actions
        self._sorted = False
        self._grouped = False
        self._sorting_mode = "ascending"
        self.y_ticks = None

    @property
    def offset(self) -> np.ndarray:
        """
        Vertical offsets applied to each visual element (e.g., for line plots).

        Returns
        -------
        np.ndarray
            Array of vertical offsets.
        """
        return self.data["offset"]

    @offset.setter
    def offset(self, values: np.ndarray) -> None:
        self.data["offset"] = values

    @property
    def scale(self) -> np.ndarray:
        """
        Scale factors applied to each visual element.

        Returns
        -------
        np.ndarray
            Array of scale multipliers.
        """
        return self.data["scale"]

    @scale.setter
    def scale(self, values: np.ndarray) -> None:
        self.data["scale"] = values

    def sort_by(self, values: dict, mode: str) -> None:
        """
        Updates the offset based on sorted group values. First row should always be at 1.

        Parameters
        ----------
        values : dict
            Mapping from index to sortable values (e.g., a metric or label).
        mode : str
            Sort direction; either 'ascending' or 'descending'.
        """
        self._sorting_mode = mode

        # Sorting items
        tmp = np.array(list(values.values()))
        unique, inverse = np.unique(tmp, return_inverse=True)
        y_order = np.argsort(unique)
        y_labels = np.sort(unique)

        if self._sorting_mode == "descending":
            y_order = len(unique) - y_order - 1
            y_labels = np.flip(y_labels)

            if self._grouped:  # Need to reverse group order
                self.data["groups"] = (
                    len(np.unique(self.data["groups"])) - self.data["groups"] - 1
                )

        order = y_order[inverse]
        self.data["order"] = order
        if self._grouped:
            self.get_offset()
            # set y ticks to unique values within each group
            y_ticks, idx = np.unique(self.data["offset"], return_index=True)
            y_labels = tmp[idx]
            self.y_ticks = {
                y_tick: y_label for y_tick, y_label in zip(y_ticks + 0.5, y_labels)
            }
        else:
            self.offset = order
            y_ticks = np.unique(order) + 0.5
            self.y_ticks = {
                y_tick: y_label for y_tick, y_label in zip(y_ticks, y_labels)
            }
        self._sorted = True

    def group_by(self, values: dict) -> None:
        """
        Updates the offset to separate elements into visual groups.

        Parameters
        ----------
        values : dict
            Mapping from index to group identifiers.
        """
        tmp = np.array(list(values.values()))
        unique, inverse = np.unique(tmp, return_inverse=True)
        groups = np.arange(len(unique))[inverse]
        y_labels = np.sort(unique)

        if self._sorted and self._sorting_mode == "descending":
            groups = len(unique) - groups - 1
            y_labels = np.flip(y_labels)

        self.data["groups"] = groups
        if self._sorted:
            self.get_offset()
            # set y ticks to middle of each group
            y_ticks = np.unique(self.data["offset"]) + 0.5
            y_ticks_groups = np.split(y_ticks, np.flatnonzero(np.diff(y_ticks) > 1) + 1)
            self.y_ticks = {
                np.mean(y_tick_group): y_label
                for y_tick_group, y_label in zip(y_ticks_groups, y_labels)
            }
        else:
            self.offset = 2 * groups
            y_ticks = np.unique(self.offset) + 0.5
            self.y_ticks = {
                y_tick: y_label for y_tick, y_label in zip(y_ticks, y_labels)
            }
        self._grouped = True

    def get_offset(self) -> None:
        order, groups = self.data["order"], self.data["groups"]
        offset = (max(order) + 1) * groups + order
        spacing = np.diff(np.hstack((-1, np.sort(offset))))
        overflow = np.where(spacing > 1, spacing - 1, 0)
        shift = np.cumsum(overflow)[offset.argsort().argsort()]
        self.offset = offset - shift + groups

    def rescale(self, factor: float) -> None:
        """
        Multiplies each element's scale by `factor`. This action is only
        possible if `sort_by` or `group_by` has been called before.

        Parameters
        ----------
        factor : float
            Scale adjustment factor (e.g., 0.1 increases scale by 10%).
        """
        if self._sorted or self._grouped:
            self.scale = self.scale + (self.scale * factor)

    def reset(self) -> None:
        """
        Resets offset and scale to default values (0 and 1 respectively).
        """
        self.data["offset"] = np.zeros(len(self.index))
        self.data["scale"] = np.ones(len(self.index))
        self._grouped = False
        self._sorted = False
        self.y_ticks = None
