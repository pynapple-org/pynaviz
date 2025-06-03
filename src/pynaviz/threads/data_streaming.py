import concurrent.futures
import threading
from numbers import Number
from typing import Callable

import pynapple as nap
import numpy as np


class TsdFrameStreaming:

    def __init__(self, data: nap.TsdFrame, callback: Callable):
        self.data = data
        self._callback = callback

        # Create underlying array for streaming
        slice_ = data.get_slice(0, 1)
        self._max_n = slice_.stop - slice_.start
        self.array = np.empty(shape=(data.shape[1], self._max_n), dtype=data.values.dtype)
        self.time = data.t[slice_]
        for i in range(data.shape[1]):
            self.array[i] = data.values[slice_,i]

    def stream(self, position, width, **kwargs) -> None:
        slice_ = self.data.get_slice(
            position[0] - width/2, position[0] + width/2
        )
        n = slice_.stop - slice_.start

        if n < self._max_n: #padding at the end
            slice_ = slice(slice_.start, slice_.stop+self._max_n-n)

        self.time = self.data.t[slice_]
        for i in range(self.data.shape[1]):
            self.array[i] = self.data.values[slice_,i]

        # Calling PlotTsdFrame _update to make sure the lines data is updated
        self._callback()

    def __len__(self):
        return self._max_n


class DataStreamingThread:

    def __init__(self, data):
        self.lock = threading.Lock()
        self.data = data

        # event that stop the loop
        self._stop_event = threading.Event()

        # create the worker
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.stream()

    def update_maps(self, time_series):
        self.request_stop()
        self.wait_until_done()
        with self.lock:
            self.color_maps = {}

        self.stream()


    def is_running(self):
        return self.future is not None and self.future.running()

    def stream(self):
        if self.is_running():
            self.request_stop()
            self.wait_until_done()
        self._stop_event.clear()
        self.future = self.worker.submit(lambda: print(1))

    def request_stop(self):
        """Request the current computation to stop."""
        self._stop_event.set()

    def wait_until_done(self, timeout=None):
        """Wait for the current future to complete."""
        if self.future:
            try:
                self.future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                pass
            except Exception as e:
                print(f"Error during mapping: {e}")

    def shutdown(self):
        self.request_stop()
        self.wait_until_done()
        self.worker.shutdown(wait=False)

    def _stream(self):
        pass