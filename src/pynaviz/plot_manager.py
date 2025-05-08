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
                "visible":np.ones(len(index), dtype=bool),
                "offset":np.zeros(len(index)),
                "scale":np.ones(len(index))
            }
        )

    @property
    def offset(self):
        return self.data['offset']

    @offset.setter
    def offset(self, values):
        self.data['offset'] = values


    def _sort_by(self, values, order):

        # First rescaling


        # Then computing offset
        tmp = np.array([v for v in values.values()])
        unique, inverse = np.unique(tmp, return_inverse=True)
        order = np.argsort(unique)
        offset = order[inverse] + 1
        self.offset += offset


    def _group_by(self, values, spacing):
        pass
