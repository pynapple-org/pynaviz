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
        """Handler for reading and decoding audio frames from a file.

        This class uses PyAV to access audio frames from a given file.
        It allows querying time-aligned audio samples between two timepoints,
        and provides audio shape and total length information.

        Parameters
        ----------
        audio_path :
            Path to the audio file.
        stream_index :
            Index of the audio stream to decode (default is 0).
        time :
            Optional 1D time axis to associate with the samples. Must match
            the number of sample points in the audio file.

        Raises
        ------
        ValueError
            If the provided `time` axis is not 1D or does not match the
            number of sample points in the audio file.
        """
        super().__init__(audio_path)
        self.stream = self.container.streams.audio[stream_index]
        self.time_base = self.stream.time_base
        self.duration = float(self.time_base * self.stream.duration)
        self.stream_index = stream_index

        self.container.seek(0)
        packet = next(self.container.demux(self.stream))
        decoded = packet.decode()
        while len(decoded) == 0:
            decoded = packet.decode()
        # initialize current frame
        self.current_frame = decoded[-1]
        self.initial_pts = self.stream.start_time
        self.pts_to_samples = int(self.current_frame.duration / self.current_frame.samples)

        # Compute tot time (this is a Fraction that can be
        # converted to float to get the duration in sec).
        self._tot_time = self.stream.time_base * self.stream.duration

        # tot_time assumes that all samples have the same length
        # but the first sometimes doesn't, here we correct for that
        nominal_length = int(self._tot_time * self.stream.rate)
        self._tot_time = float(self._tot_time)
        full_frame_size = self.stream.codec_context.frame_size
        self._tot_samples = nominal_length - full_frame_size + self.current_frame.samples

        self._time = self._check_and_cast_time(time)
        self._initial_experimental_time_sec = 0 if self._time is None else self._time[0]

    def _check_and_cast_time(self, time):
        if time is not None and len(time) != self._tot_samples:
            raise ValueError(
                "The provided time axis doesn't match the number of sample points in the audio file.\n"
                f"Actual number of sample points: {self._tot_samples}\n"
                f"Provided number of sample points: {len(time)}"
            )
        if time is not None:
            time = np.array(time, dtype=float)
            if time.ndim > 1:
                raise ValueError(f"'time' must be 1 dimensional. {time.ndim}-array provided.")
        return time

    def ts_to_pts(self, ts: float) -> int:
        """
        Convert time point to sound pts, clipping if necessary.

        Parameters
        ----------
        ts :
            Experimental timestamp to match.

        Returns
        -------
        idx :
            Index of the frame with time <= `ts`. Clipped to [0, len(time) - 1].
        """
        ts = np.clip(ts - self._initial_experimental_time_sec, 0, self.duration)
        return int(ts  / self.time_base + self.stream.start_time)

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
        """
        Extract a chunk of audio samples between two timestamps.

        Parameters
        ----------
        start :
            Start time in seconds.
        end :
            End time in seconds.

        Returns
        -------
        :
            2D array of shape (num_samples, num_channels) containing the audio data.
        """
        start = self.ts_to_pts(start)
        end = self.ts_to_pts(end)

        current_pts = start

        if self._need_seek_call(getattr(self.current_frame, "pts", None),
                                start):
            self.container.seek(
                start,
                backward=True,
                any_frame=False,
                stream=self.stream,
            )
            self.current_frame = None

        frames: List[NDArray] = []
        first_frame = True

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

                if first_frame:
                    idx = (start - frame.pts) // self.pts_to_samples
                    frames.append(frame.to_ndarray()[:, idx:])
                    first_frame = False
                else:
                    frames.append(frame.to_ndarray())
                current_pts = frame.pts + frame.duration
                self.current_frame = frame

        idx = (frame.pts + frame.duration - end) // self.pts_to_samples
        frames = np.concatenate(frames, axis=1).T
        # handle corner case: end == frame.pts + frame.duration
        frames = frames[:-idx] if idx > 0 else frames
        return frames

    @property
    def time(self):
        """
        Time axis corresponding to the audio samples.

        Returns
        -------
        :
            Array of timestamps with shape (num_samples,).
        """
        # generate time on the fly or use provided
        return (
            self._time if self._time is not None else
            np.linspace(0, float(self._tot_time), self._tot_samples)
        )
    @time.setter
    def time(self, time):
        self._time = self._check_and_cast_time(time)

    @property
    def shape(self):
        """
        Shape of the audio data.

        Returns
        -------
        :
            Tuple `(num_samples, num_channels)` describing the audio shape.
        """
        return self._tot_samples, self.stream.codec_context.channels

    @property
    def tot_length(self):
        """
        Total duration of the audio in seconds.

        Returns
        -------
        :
            Total duration of the audio stream.
        """
        return self._tot_time
