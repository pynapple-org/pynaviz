"""
Base class for audio and video handling.

Handles opening/closing the stream, seeking, and keyframe extraction.
"""

import abc
import pathlib
import threading

import av
import numpy as np

# from line_profiler import profile


class BaseAudioVideo:
    _thread_local = threading.local()

    def __init__(
        self,
        path: str | pathlib.Path,
    ) -> None:
        self._thread_local.get_from_index = False
        self.file_path = pathlib.Path(path)
        self.container = av.open(path)
        self._running = True

        # initialize index for last decoded frame
        # if sampling of other signals (LFP) is much denser, multiple times the frame
        # is unchanged, so cache the idx
        self.last_loaded_idx = None

        self._lock = threading.Lock()

        self._keyframe_pts = []
        self._pts_keyframe_ready = threading.Event()
        self._keyframe_thread = threading.Thread(target=self._extract_keyframes_pts, daemon=True)
        self._keyframe_thread.start()

    @abc.abstractmethod
    def _ts_to_pts(self, ts: float) -> int:
        pass

    @abc.abstractmethod
    def _extract_keyframes_pts(self):
        pass

    def _need_seek_call(self, current_frame_pts, target_frame_pts):
        if current_frame_pts is None:
            return True

        with self._lock:
            if len(self._keyframe_pts) == 0:
                return True
            # While the keyframe thread is still running we may not yet know
            # about a keyframe that sits between current position and target.
            # Seek conservatively so we don't miss it.
            # Once the thread is done the list is complete: no keyframe beyond
            # the last known one exists, so the absence of one is not a reason
            # to seek — we can stream forward safely.
            if not self._pts_keyframe_ready.is_set() and self._keyframe_pts[-1] < target_frame_pts:
                return True

        # roll back the stream if audiovideo is scrolled backwards
        if current_frame_pts > target_frame_pts:
            return True

        # find the closest keyframe pts before a given frame
        idx = np.searchsorted(self._keyframe_pts, target_frame_pts, side="right")
        closest_keyframe_pts = self._keyframe_pts[max(0, idx - 1)]

        # seek forward only if there is a keyframe between current position
        # and the target (i.e. a closer starting point exists).
        return closest_keyframe_pts > current_frame_pts


    def close(self):
        """Close the audiovideo stream."""
        self._running = False
        threads = ["_index_thread", "_keyframe_thread"]
        for thread_name in threads:
            # index thread is only for video frames
            thread = getattr(self, thread_name, None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=1)
        try:
            self.container.close()
        except Exception:
            print("AudioHandler failed to close the audiovideo stream.")
        finally:
            # dropping refs to fully close av.InputContainer
            self.container = None
            self.stream = None

    # context protocol
    # (with AudioHandler(path) as audiovideo ensure closing)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
