import numpy as np
from collections import UserDict

class PlotManager(UserDict):
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
        super().__init__(
            {"groups":np.zeros(len(index)),
             "order":np.arange(0, len(index)),
             "visible":np.ones(len(index), dtype=bool)
             }
        )
        self.index = index

    @property
    def groups(self):
        return {k:v for k,v in zip(self.index,self['groups'])}

    @groups.setter
    def groups(self, value):
        self["groups"] = value

    @property
    def order(self):
        return {k:v for k,v in zip(self.index,self['order'])}

    @order.setter
    def order(self, value):
        self["order"] = value

    @property
    def visible(self):
        return {k:v for k,v in zip(self.index,self['visible'])}

    @visible.setter
    def visible(self, value):
        self["visible"] = value

    def __getitem__(self, item):
        """

        Parameters
        ----------
        item:
            - str: either ["groups", "order", "visible"]
            - Number o: metadata index (for TsGroup and IntervalSet)
            - list, np.ndarray, pd.Series: metadata index (for TsGroup and IntervalSet)
            - tuple: metadata index and column name (for TsGroup and IntervalSet)


        Returns
        -------

        """

    def __setitem__(self, key, value):
        pass