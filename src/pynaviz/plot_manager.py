import numpy as np
from pynapple.core.metadata_class import _Metadata


class _PlotManager:
    """
    Class that keep track of the actions applied to the visual. The information that
    are being tracked are :
        - `groups` from action group_by.
        - `order` from action sort_by.
        - `visible` from selection action.
    """

    def __init__(self, index):
        """
        By default, passing just `index` initialize the PlotManager
        with default values.

        Parameters
        ----------
        index: list or np.ndarray
            The index of the elements. Either index in TsGroup, column labels
            in TsdFrame or index in IntervalSet.
        """
        self.index = index
        self.data = _Metadata(
            index = index,
            data = {
                "groups":np.zeros(len(index), dtype="int"),
                "order":np.arange(0, len(index)),
                "visible":np.ones(len(index), dtype=bool)
            }
        )

    def _sorted_y_pos(self, values, order):
        # Need to get each groups
        import pandas as pd
        values = pd.Series(values)
        idx_sorted = values.sort_values(ascending=(order == order))
        return {idx: i for i, idx in enumerate(idx_sorted.index)}

    def _grouped_y_pos(self, values, spacing):
        pass
