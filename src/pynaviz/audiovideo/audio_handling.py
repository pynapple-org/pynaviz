from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import av
import numpy as np

# from line_profiler import profile
from numpy.typing import NDArray

from .base_audiovideo import BaseAudioVideo


class AudioHandler(BaseAudioVideo):
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

    Examples
    --------
    >>> from pynaviz.audiovideo import AudioHandler
    >>> ah = AudioHandler("example.mp3")  # doctest: +SKIP
    >>> # Get audio samples between 1.5 and 2.5 seconds.
    >>> audio_trace = ah.get(1.5, 2.5)  # doctest: +SKIP
    >>> # Shape: (n_samples, n_channels)
    >>> audio_trace.shape  # doctest: +SKIP
    (44100, 2)
    """

    def __init__(
        self,
        audio_path: str | pathlib.Path,
        stream_index: int = 0,
        time: Optional[NDArray] = None,
    ) -> None:

        super().__init__(audio_path)
        self.stream = self.container.streams.audio[stream_index]
        self.time_base = self.stream.time_base
        self.stream_index = stream_index

        self.container.seek(0)
        packet = next(self.container.demux(self.stream))
        decoded = packet.decode()
        while len(decoded) == 0:
            decoded = packet.decode()

        # initialize current frame
        self.current_frame = decoded[-1]
        self.start_time_pts = self.stream.start_time if self.stream.start_time is not None else self.current_frame.pts
        self.stop_time_pts = self.start_time_pts + self.stream.duration
        self.pts_to_samples = int(self.current_frame.duration / self.current_frame.samples)

        self._tot_samples = int(
            self.stream.time_base * self.stream.duration * self.stream.rate
        )
        self.duration = float(self._tot_samples / self.stream.rate)
        # if the codec context stores the frame_size info, then get the size form it,
        # otherwise assume that each frame has the same size (as in a wav)
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

    def _ts_to_pts(self, ts: float) -> int:
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
        return int(ts / self.time_base + self.start_time_pts)

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

    def _decode_first(self, start: int):
        """Decode first frame backing off if more warm-up is needed.

        This method attempts to start decoding at `start`, handling formats
        (notably MP3) that may require a preroll due to inter-frame dependencies.

        **Behavior:**
        - If the current frame in memory already covers `start`, return a
          sliced version without seeking.
        - For most formats (WAV, FLAC, etc.), decoding starts exactly at `start`.
        - For MP3, decoding may start slightly earlier (initially 100 ms before
          `start`) to avoid returning an empty or silent first frame. If the
          first decoded frame is invalid, the preroll is increased in 100 ms
          steps, up to 1 second.

        Parameters
        ----------
        start : int
            Start position in presentation timestamp (PTS) units.

        Returns
        -------
        frames : list of np.ndarray
            Decoded audio data for the first chunk starting at `start`
            (possibly beginning earlier for MP3 preroll).
        current_pts : int
            PTS of the last decoded frame.

        Raises
        ------
        ValueError
            If no valid first frame can be decoded within the allowed preroll
            (for MP3) or starting exactly at `start` (for other formats).

        Notes
        -----
        MP3 decoding requires preroll because frames are not independently
        decodable; starting too close to `start` can yield an all-zero frame.
        Other formats are decoded directly from `start`.
        """
        # Fast-path: current frame already spans 'start'
        cf = self.current_frame
        if cf is not None and cf.pts is not None and cf.pts <= start <= cf.pts + cf.duration:
            idx = (start - cf.pts) // self.pts_to_samples
            return [cf.to_ndarray()[:, idx:]], cf.pts + cf.duration

        # change this if more than one format needs a preroll
        is_mp3 = self.container.format.name == "mp3"
        max_backoff = int(1.0 / self.time_base)  # ~1s in PTS
        backoff_step = int(0.1 / self.time_base)  # ~100ms in PTS

        # MP3 typically needs preroll; others generally don't
        backoff = backoff_step if is_mp3 else 0

        while (start - max(0, start - backoff)) < max_backoff:
            safe_start = max(0, start - backoff)

            # Seek once per attempt
            self.container.seek(
                safe_start,
                backward=True,
                any_frame=False,
                stream=self.stream,
            )
            self.current_frame = None

            current_pts = safe_start
            first_frame = True
            frames: list[np.ndarray] = []
            need_retry_with_more_backoff = False

            # Demux packets until we either assemble the first slice or decide to retry
            for packet in self.container.demux(self.stream):
                decoded = packet.decode()
                if not decoded:
                    continue

                for frame in decoded:
                    if frame.pts is None:
                        continue

                    # Skip warmup frames before 'start'
                    if frame.pts + frame.duration <= start:
                        self.current_frame = frame
                        current_pts = frame.pts + frame.duration
                        continue

                    # We’re at the frame covering 'start'
                    arr = frame.to_ndarray()

                    # MP3: first frame after seek can decode as silence (zeros); back off and retry
                    # Note that mp3 returns all zeros if frame was iif not decoded properly. This may
                    # lead to erroneous conclusion for non-mp3s.
                    if is_mp3 and first_frame and not np.any(arr):
                        need_retry_with_more_backoff = True
                        break  # break inner frame loop to increase backoff and re-seek

                    if first_frame:
                        idx = (start - frame.pts) // self.pts_to_samples
                        frames.append(arr[:, idx:])
                        first_frame = False
                    else:
                        frames.append(arr)

                    current_pts = frame.pts + frame.duration
                    self.current_frame = frame

                # Decide what to do after processing this packet
                if need_retry_with_more_backoff:
                    break  # go increase backoff and retry a new attempt

                if frames:
                    # We’ve produced the first slice; return to let caller continue decoding
                    return frames, current_pts

                # else: no usable frame yet → continue demuxing next packet

            if need_retry_with_more_backoff:
                backoff += backoff_step
                continue


        raise ValueError("Failed to decode the first audio frame with sufficient preroll.")

    def get(self, start: float, end: float) -> NDArray:
        """
        Extract decoded frames from a video between two timestamps.

        This method decodes and returns the raw video frames corresponding to
        the time interval ``[start, end]`` in seconds. Decoding begins from the
        nearest keyframe at or before ``start``, and proceeds sequentially until
        the end timestamp is reached or exceeded. If the last decoded frame
        extends beyond ``end``, trailing samples are truncated so that the
        returned array aligns with the requested time range.

        Parameters
        ----------
        start : float
            Start time of the segment to extract, in seconds.
        end : float
            End time of the segment to extract, in seconds. Must be greater
            than ``start``.

        Returns
        -------
        :
            A 2D NumPy array containing the decoded frames for the requested
            interval. The exact shape depends on the video format and frame size,
            with the first dimension corresponding to time (frame index or
            samples) and the remaining dimensions containing
            audio channels.

        Notes
        -----
        - The returned frames are decoded in sequence and concatenated before
          being transposed so that time is the first dimension.
        - If ``end`` falls between two frames, the last frame is partially trimmed
          to match the requested duration.

        See Also
        --------
        `av.AudioFrame <https://pyav.org/docs/stable/api/audio.html#module-av.audio.frame>`
            The PyAV frame object.
        """
        if end < start:
            raise ValueError(f"`end` time must be greater than `start` time. "
                             f"Provided start is {start}, and provided end is {end}.")
        elif start > self.duration:
            return np.zeros((0, self.stream.codec_context.channels), dtype=np.float32)

        start_pts = self._ts_to_pts(start)
        end_pts = self._ts_to_pts(end)

        frames, current_pts = self._decode_first(start_pts)

        # Decode subsequent full frames until we pass 'end'
        for packet in self.container.demux(self.stream):
            decoded = packet.decode()
            if not decoded:
                continue

            for frame in decoded:
                if frame.pts is None:
                    continue

                frames.append(frame.to_ndarray())
                current_pts = frame.pts + frame.duration
                self.current_frame = frame

                if current_pts >= end_pts:
                    break
            if current_pts >= end_pts:
                break

        # Concatenate and trim tail if we overshot 'end'
        out = np.concatenate(frames, axis=1).T

        overhang_pts = current_pts - end_pts
        if overhang_pts > 0:
            chop = int(overhang_pts // self.pts_to_samples)
            if chop > 0:
                out = out[:-chop]

        return out

    @property
    def t(self) -> NDArray:
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
            np.linspace(0, float(self.duration), self._tot_samples)
        )
    @t.setter
    def t(self, time):
        self._time = self._check_and_cast_time(time)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Shape of the audio data.

        Returns
        -------
        :
            Tuple `(num_samples, num_channels)` describing the audio shape.
        """
        return self._tot_samples, self.stream.codec_context.channels

    @property
    def tot_length(self) -> float:
        """
        Total duration of the audio in seconds.

        Returns
        -------
        :
            Total duration of the audio stream.
        """
        return self.duration

    def __len__(self):
        return self._tot_samples

    @property
    def index(self) -> NDArray:
        """
        Time axis corresponding to the audio samples.

        Returns
        -------
        :
            Array of timestamps with shape (num_samples,).
        """
        return self.t
