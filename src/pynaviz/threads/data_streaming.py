import concurrent.futures
import threading
from numbers import Number
import pynapple as nap
import numpy as np


class TsdFrameStreaming:

    def __init__(self, data: nap.TsdFrame):
        self.data = data

    def __getitem__(self, key):
        """Should yield a 3D array of float32 for each line object"""
        pass

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