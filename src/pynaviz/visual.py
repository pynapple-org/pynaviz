from typing import Tuple

import fastplotlib as fpl


class Visual:
    def __init__(
            self
    ):
        self._figure = fpl.Figure()

    @property
    def figure(self) -> fpl.Figure:
        return self._figure
