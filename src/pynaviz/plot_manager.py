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
                "order": np.arange(0, len(index)),
                "visible": np.ones(len(index), dtype=bool),
                "offset": np.zeros(len(index)),
                "scale": np.ones(len(index)),
            }
        )
        # To keep track of past actions
        self._sorted = False
        self._grouped = False

    @property
    def offset(self) -> np.ndarray:
        """
        Vertical offsets applied to each visual element (e.g., for line plots).

        Returns
        -------
        np.ndarray
            Array of vertical offsets.
        """
        return self.data['offset']

    @offset.setter
    def offset(self, values: np.ndarray) -> None:
        self.data['offset'] = values

    @property
    def scale(self) -> np.ndarray:
        """
        Scale factors applied to each visual element.

        Returns
        -------
        np.ndarray
            Array of scale multipliers.
        """
        return self.data['scale']

    @scale.setter
    def scale(self, values: np.ndarray) -> None:
        self.data['scale'] = values

    def sort_by(self, values: dict, order: str) -> None:
        """
        Updates the offset based on sorted group values.

        Parameters
        ----------
        values : dict
            Mapping from index to sortable values (e.g., a metric or label).
        order : str
            Sort direction; either 'ascending' or 'descending'.
        """
        tmp = np.array(list(values.values()))
        unique, inverse = np.unique(tmp, return_inverse=True)
        y_order = np.argsort(unique)
        offset = y_order[inverse] + 1
        if order == "descending":
            offset = offset[::-1]
        self.offset += offset
        self._sorted = True

    def group_by(self, values: dict, spacing: int) -> None:
        """
        Updates the offset to separate elements into visual groups.

        Parameters
        ----------
        values : dict
            Mapping from index to group identifiers.
        spacing : int
            Unused parameter (reserved for future logic).
        """
        tmp = np.array(list(values.values()))
        unique, inverse = np.unique(tmp, return_inverse=True)
        offset = np.arange(1, len(unique) + 1)[inverse]
        self.offset += offset
        self._grouped = True

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
