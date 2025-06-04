from typing import Callable

import pynapple as nap
import numpy as np


class TsdFrameStreaming:

    def __init__(self, data: nap.TsdFrame, callback: Callable):
        self.data = data
        self._callback = callback
        self.max = np.max(data, axis=0)
        self.min = np.min(data, axis=0)

        # Create underlying array for streaming
        slice_ = data._get_slice(-10, 11)
        self._max_n = slice_.stop - slice_.start
        self.array = np.empty(shape=(data.shape[1], self._max_n), dtype=data.values.dtype)
        self.time = data.t[slice_]
        for i in range(data.shape[1]):
            self.array[i] = data.values[slice_,i]

    def _flush(self, slice_):
        self.time = self.data.t[slice_]
        for i in range(self.data.shape[1]):
            self.array[i] = self.data.values[slice_, i]
        self._callback()

    def stream(self, position, width, **kwargs) -> None:
        slice_ = self.data._get_slice(
            position[0] - width/2 - 10, position[0] + width/2 + 10, n_points=int(self._max_n)
        )

        if slice_.step is not None and slice_.step > 1:
            # zooming out
            self._flush(slice_)
        elif (slice_.step is None or slice_.step == 1) and (slice_.stop - slice_.start) == self._max_n:
            # Panning
            self._flush(slice_)
        else:
            pass

    def __len__(self):
        return self._max_n


