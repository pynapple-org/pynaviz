import pathlib
import threading
import time
import warnings
from contextlib import contextmanager
from typing import List, Optional, Tuple

import av
import numpy as np

# from line_profiler import profile
from numpy.typing import NDArray


class AudioHandler:
    """Class for getting audiovideo frames."""

    _thread_local = threading.local()

    def __init__(
        self,
        audio_path: str | pathlib.Path,
        stream_index: int = 0,
        time: Optional[NDArray] = None,
        return_frame_array: bool = True,
    ) -> None:
        self._thread_local.get_from_index = False
        self.audio_path = pathlib.Path(audio_path)
        self.container = av.open(audio_path)
        self.stream = self.container.streams.audio[stream_index]
        self.time_base = self.stream.time_base
        self.duration = float(self.time_base * self.stream.duration)
        self.stream_index = stream_index
        self.return_frame_array = return_frame_array
        self._running = True

        # initialize index for last decoded frame
        # if sampling of other signals (LFP) is much denser, multiple times the frame
        # is unchanged, so cache the idx
        self.last_loaded_idx = None

        # initialize current frame
        self.current_frame: Optional[av.AudioFrame] = None
        self.current_frame_pts: Optional[tuple[int, int]] = None

        self._lock = threading.Lock()

        self._keyframe_pts = []
        self._pts_keyframe_ready = threading.Event()
        self._keyframe_thread = threading.Thread(target=self._extract_keyframes_pts, daemon=True)
        self._keyframe_thread.start()

    def ts_to_pts(self, ts: float) -> int:
        """
        Convert time point to sound pts, clipping if necessary.

        Parameters
        ----------
        ts : float
            Experimental timestamp to match.

        Returns
        -------
        idx : int
            Index of the frame with time <= `ts`. Clipped to [0, len(time) - 1].
        """
        ts = np.clip(ts, 0, self.duration)
        return int(ts / self.time_base)

    def _extract_keyframes_pts(self):
        try:
            with av.open(self.audio_path) as container:
                stream = container.streams.audio[0]
                for packet in container.demux(stream):
                    if not self._running:
                        return
                    if packet.is_keyframe:
                        with self._lock:
                            self._keyframe_pts.append(packet.pts)
        except Exception as e:
            # do not block gui
            print("Keyframe thread error:", e)
        finally:
            with self._lock:
                self._keyframe_pts = np.asarray(self._keyframe_pts)
            self._pts_keyframe_ready.set()

    def _need_seek_call(self, current_frame_pts, target_frame_pts):
        with self._lock:
            # return if empty list or empty array or not enough frmae
            if len(self._keyframe_pts) == 0 or self._keyframe_pts[-1] < target_frame_pts:
                return True

        # roll back the stream if audiovideo is scrolled backwards
        if current_frame_pts > target_frame_pts:
            return True

        # find the closest keyframe pts before a given frame
        idx = np.searchsorted(self._keyframe_pts, target_frame_pts, side="right")
        closest_keyframe_pts = self._keyframe_pts[max(0, idx - 1)]

        # if target_frame_pts is larger than current (and if code
        # arrives here, it is, see second return statement),
        # then seek forward if there is a future keyframe closest
        # to the target.
        return closest_keyframe_pts > current_frame_pts

    def close(self):
        """Close the audiovideo stream."""
        self._running = False
        if self._index_thread.is_alive():
            self._index_thread.join(timeout=1)  # Be conservative, don’t block forever
        if self._keyframe_thread.is_alive():
            self._keyframe_thread.join(timeout=1)
        try:
            self.container.close()
        except Exception:
            print("AudioHandler failed to close the audiovideo stream.")
        finally:
            # dropping refs to fully close av.InputContainer
            self.container = None
            self.stream = None

    # SHIFTED BY SOME WEIRD AMOUNT (~50 samples)
    def get(
        self,
        start: float,
        end: float,
    ) -> Tuple[int, List[av.AudioFrame] | NDArray, av.AudioFrame]:

        start = self.ts_to_pts(start)
        end = self.ts_to_pts(end)

        current_pts = start

        self.container.seek(
            start,
            backward=True,
            any_frame=False,
            stream=self.stream,
        )
        self.current_frame = None
        preceding_frame = None
        go_to_next_packet = True

        frames = []

        while current_pts < end:
            packet = next(self.container.demux(self.stream))

            try:
                decoded = packet.decode()
                while len(decoded) == 0:
                    decoded = packet.decode()
            except av.error.EOFError:
                # end of the audiovideo, rewind
                break

            for frame in decoded:
                if frame.pts is None:
                    continue

                found_next = frame.pts > current_pts
                found_current = frame.pts == current_pts or (found_next and preceding_frame is None)

                if found_current:
                    frames.append(frame)
                    go_to_next_packet = False
                elif found_next:
                    frames.append(preceding_frame)
                    go_to_next_packet = False
                else:
                    go_to_next_packet = True

                preceding_frame = frame
                current_pts = frames[-1].pts

        return np.concatenate([f.to_ndarray() for f in frames], axis=1).T

    # context protocol
    # (with AudioHandler(path) as audiovideo ensure closing)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
