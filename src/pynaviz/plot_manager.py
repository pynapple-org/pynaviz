"""
Plot manager for TsGroup, TsdFrame, and IntervalSet visualizations.
The manager gives context for which action has been applied to the visual.
"""
from typing import TYPE_CHECKING

import numpy as np
from pynapple.core.metadata_class import _Metadata

if TYPE_CHECKING:
    from .base_plot import _BasePlot


def to_python_type(val):
    if hasattr(val, 'item'):  # numpy scalar
        return val.item()
    elif hasattr(val, 'tolist'):
        return val.tolist()
    else:
        return val


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

    def __init__(self, index: list | np.ndarray, base_plot: "_BasePlot"):
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
                "offset": base_plot._initialize_offset(index),
                "scale": np.ones(len(index)),
            },
        )

        # To keep track of past actions
        self._actions: dict[str, None | dict] = {
            "group_by": None,
            "sort_by": None,
            "color_by": None,
        }
        self.y_ticks = None

    @property
    def is_sorted(self) -> bool:
        return self._actions["sort_by"] is not None

    @property
    def is_grouped(self) -> bool:
        return self._actions["group_by"] is not None

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

    def sort_by(self, values: dict, metadata_name: str, mode: str) -> None:
        """
        Updates the offset based on sorted group values. First row should always be at 1.

        Parameters
        ----------
        values : dict
            Mapping from index to sortable values (e.g., a metric or label).
        mode : str
            Sort direction; either 'ascending' or 'descending'.
        """
        was_grouped = self.is_grouped

        # Sorting items
        tmp = np.array(list(values.values()))
        unique, inverse = np.unique(tmp, return_inverse=True)
        y_order = np.argsort(unique)
        y_labels = np.sort(unique)

        if mode == "descending":
            y_order = len(unique) - y_order - 1
            y_labels = np.flip(y_labels)

            if was_grouped:  # Need to reverse group order
                self.data["groups"] = (
                    len(np.unique(self.data["groups"])) - self.data["groups"] - 1
                )

        order = y_order[inverse]
        self.data["order"] = order
        if was_grouped:
            self.set_offset()
            # set y ticks to unique values within each group
            y_ticks, idx = np.unique(self.data["offset"], return_index=True)
            y_labels = tmp[idx]
            self.y_ticks = {
                y_tick: y_label for y_tick, y_label in zip(y_ticks, y_labels)
            }
        else:
            self.offset = order
            y_ticks = np.unique(order)
            self.y_ticks = {
                y_tick: y_label for y_tick, y_label in zip(y_ticks, y_labels)
            }

        # action dictionary must store the action inputs
        self._actions["sort_by"] = dict(metadata_name=metadata_name, mode=mode)

    def group_by(self, values: dict, metadata_name:str, **kwargs) -> None:
        """
        Updates the offset to separate elements into visual groups.

        Parameters
        ----------
        values : dict
            Mapping from index to group identifiers.
        metadata_name :
            Name of the metadata column used for grouping.
        """
        tmp = np.array(list(values.values()))
        unique, inverse = np.unique(tmp, return_inverse=True)
        groups = np.arange(len(unique))[inverse]
        y_labels = np.sort(unique)

        # check previous configs
        was_sorted = self.is_sorted
        was_descending = self._actions["sort_by"]["mode"] == "descending" if was_sorted else False

        if was_sorted and was_descending:
            groups = len(unique) - groups - 1
            y_labels = np.flip(y_labels)

        self.data["groups"] = groups
        if was_sorted:
            self.set_offset()
            # set y ticks to middle of each group
            y_ticks = np.unique(self.data["offset"])
            y_ticks_groups = np.split(y_ticks, np.flatnonzero(np.diff(y_ticks) > 1) + 1)
            self.y_ticks = {
                np.mean(y_tick_group): y_label
                for y_tick_group, y_label in zip(y_ticks_groups, y_labels)
            }
        else:
            self.offset = 2 * groups
            y_ticks = np.unique(self.offset)
            self.y_ticks = {
                y_tick: y_label for y_tick, y_label in zip(y_ticks, y_labels)
            }
        self._actions["group_by"] = dict(metadata_name=metadata_name, **kwargs)

    def color_by(self, values: None, metadata_name:str, **kwargs) -> None:
        """Store the color_by action parameters.

        Notes
        -----
        `values` is unused but kept for consistency for all methods. We may think of
        implementing the setter logic here for the colors, which would require a
        slightly different logic for the different _BasePlot concrete implementations.
        """
        self._actions["color_by"] = dict(metadata_name=metadata_name, **kwargs)

    def set_offset(self) -> None:
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
        was_grouped = self.is_grouped
        was_sorted = self.is_sorted
        if was_grouped or was_sorted:
            self.scale = self.scale + (self.scale * factor)

    def reset(self) -> None:
        """
        Resets offset and scale to default values (0 and 1 respectively).
        """
        self.data["offset"] = np.zeros(len(self.index))
        self.data["scale"] = np.ones(len(self.index))
        self.y_ticks = None
        self._actions = {
            "group_by": None,
            "sort_by": None,
            "color_by": None,
        }

    def get_state(self) -> dict:
        """
        Returns the current state of the plot manager in a serializable format.

        Returns
        -------
        dict
            Dictionary containing all information needed to restore the current
            visual state, including sorting, grouping, scaling, and visibility.
        """
        if hasattr(self.index, "tolist"):
            index = self.index.tolist()
        else:
            index = list(self.index)
        if self.y_ticks is not None:
            serializable_y_ticks = {to_python_type(k): to_python_type(v) for k, v in self.y_ticks.items()}
        else:
            serializable_y_ticks = None
        state = {
            'index': index,  # Convert to list for JSON serialization
            'groups': self.data['groups'].tolist(),
            'order': self.data['order'].tolist(),
            'visible': self.data['visible'].tolist(),
            'offset': self.data['offset'].tolist(),
            'scale': self.data['scale'].tolist(),
            'y_ticks': serializable_y_ticks,
        }
        serializable_actions = {
            action: {
                k: v.tolist() if hasattr(v, "tolist()") else v
                for k, v in kwargs.items()
            } if kwargs is not None else None
            for action, kwargs in self._actions.items()
        }
        state.update(dict(_actions=serializable_actions))
        return state

    @classmethod
    def from_state(cls, base_plot: "_BasePlot", state: dict, index: list | None=None) -> '_PlotManager':
        """
        Creates a new PlotManager instance from a previously saved state.

        Parameters
        ----------
        base_plot:
            The _BasePlot object that applies the action.
        state : dict
            Dictionary containing the saved state from get_state() method.
        index :
            The index of the plot manager to use. If None, assume that the index
            is unchanged, so use the stored one.

        Returns
        -------
        _PlotManager
            New instance with the restored state.
        """
        # Create instance with the saved index & order etc.
        if index is None:
            instance = cls(state["index"], base_plot)
            # Restore all data arrays
            instance.data['groups'] = np.array(state['groups'], dtype=int)
            instance.data['order'] = np.array(state['order'], dtype=int)
            instance.data['visible'] = np.array(state['visible'], dtype=bool)
            instance.data['offset'] = np.array(state['offset'])
            instance.data['scale'] = np.array(state['scale'])
            # Restore other params & actions configs
            instance.y_ticks = state['y_ticks']
            instance._actions = state['_actions']
        else:
            instance = cls(index, base_plot)
            # replace instance
            base_plot._manager = instance
            for action, kwargs in state['_actions'].items():
                attr = getattr(base_plot, action, None)
                if kwargs is not None and attr is not None:
                    attr(**kwargs)
        return instance

