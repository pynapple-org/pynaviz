import pathlib
from typing import List, Optional

import av
import numpy as np

# from line_profiler import profile
from numpy.typing import NDArray

from .base_audiovideo import BaseAudioVideo


class AudioHandler(BaseAudioVideo):
    """Class for getting audiovideo frames."""


    def __init__(
        self,
        audio_path: str | pathlib.Path,
        stream_index: int = 0,
        time: Optional[NDArray] = None,
    ) -> None:
        super().__init__(audio_path)
        self.stream = self.container.streams.audio[stream_index]
        self.time_base = self.stream.time_base
        self.duration = float(self.time_base * self.stream.duration)
        self.stream_index = stream_index

        # initialize current frame
        self.current_frame: Optional[av.AudioFrame] = None


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
            with av.open(self.file_path) as container:
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

    def get(
        self,
        start: float,
        end: float,
    ) -> NDArray:

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

        frames: List[NDArray] = []

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

                frames.append(frame.to_ndarray())
                current_pts = frame.pts

        return np.concatenate(frames, axis=1).T
