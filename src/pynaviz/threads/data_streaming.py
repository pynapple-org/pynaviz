from typing import Callable

import pynapple as nap


class TsdFrameStreaming:
    """
    A class for streaming fixed-size windows of a `nap.TsdFrame` to a callback function,
    based on a desired position and zoom level.

    This is useful for building interactive, time-based visualizations where the window
    content is updated as the user pans or zooms over time.

    Attributes
    ----------
    data : nap.TsdFrame
        The time series data to stream.
    _callback : Callable
        A function that receives a slice object indicating the time window to display.
    window_size : float
        The size of the time window (in same units as TsdFrame timestamps).
    """

    def __init__(self, data: nap.TsdFrame, callback: Callable[[slice], None], window_size: float):
        """
        Initialize the TsdFrameStreaming object.

        Parameters
        ----------
        data : nap.TsdFrame
            The input time series data to stream.
        callback : Callable[[slice], None]
            A function to be called with the computed slice when streaming.
        window_size : float
            The time duration (in same units as data timestamps) of the streaming window.
        """
        self.data = data
        self._callback = callback
        self.window_size = window_size

        # Determine how many points fall in a window of size `window_size`
        # This assume that data is uniformly sampled
        self._slice_ = data._get_slice(data.t[0], data.t[0] + window_size)
        self._max_n = (self._slice_.stop - self._slice_.start) + 1

        self._flushed = False # To keep tracked if the data has been flushed at least once

    def get_slice(self, start: float, end: float) -> slice:
        """
        Compute a slice centered around the requested window, extended to match internal resolution.

        Parameters
        ----------
        start : float
            Start time of the requested display window.
        end : float
            End time of the requested display window.

        Returns
        -------
        slice
            A slice object corresponding to the adjusted time window.
        """
        if start >= end:
            raise ValueError("Start time must be less than end time.")

        width = end - start

        slice_ = self.data._get_slice(
            start - width,
            end + width,
            n_points=int(self._max_n)
        )

        return slice_

    def stream(self, position: tuple, width: float, **kwargs) -> None:
        """
        Stream a slice of data to the callback based on the current position and zoom level.

        Parameters
        ----------
        position : (float, float)
            Center time position of the requested view window in xy coordinates. This is a tuple (x, y),
            but only the x-coordinate (time) is used for slicing.
        width : float
            Width of the requested view window.
        **kwargs :
            Additional arguments passed to the callback (not used in this base class).
        """
        new_slice_ = self.get_slice(position[0] - width / 2, position[0] + width / 2)

        # print(self._slice_, new_slice_)

        if new_slice_ == self._slice_:
            if not self._flushed:
                self._flushed = True
                self._callback(new_slice_)
            return
        else:
            if new_slice_.start > self._slice_.start and new_slice_.stop < self._slice_.stop:
                self._flushed = False
                return  # zooming in, ignore for now
            else:
                self._callback(new_slice_)
                self._slice_ = new_slice_
                self._flushed = True

        # # Edge cases
        # if slice_.start == 0 or slice_.stop == self.data.shape[0]:
        #     self._callback(slice_)
        #
        # if slice_.step is not None and slice_.step > 1:
        #     # Zooming out — reduced resolution
        #     self._callback(slice_)
        # elif (slice_.step is None or slice_.step == 1) and (slice_.stop - slice_.start) == self._max_n:
        #     # Panning — full-resolution window
        #     self._callback(slice_)
        # else:
        #     # Zooming in — resolution higher than base window, currently ignored
        #     pass

    def __len__(self) -> int:
        """
        Return the number of data points in a base-resolution window.

        Returns
        -------
        int
            Number of samples in the window (self._max_n).
        """
        return self._max_n
