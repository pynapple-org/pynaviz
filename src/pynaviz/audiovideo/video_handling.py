from __future__ import annotations

import pathlib
import threading
import time
import warnings
from collections import deque
from contextlib import contextmanager
from typing import List, Optional, Tuple

import av
import numpy as np

# from line_profiler import profile
from numpy.typing import NDArray

from .base_audiovideo import BaseAudioVideo

# Number of packets to buffer before flushing to the index for codecs without
# B-frames (where packet PTS are already in display order).
_INDEX_FLUSH_EVERY = 64


def _needs_flush(count_keyframes: int, temp: list, has_b_frames: bool, n_b_frames: int = 1) -> bool:
    """True when the buffered GOP / batch is ready to commit to the index."""
    if has_b_frames:
        return (count_keyframes == n_b_frames) and bool(temp)
    return len(temp) >= _INDEX_FLUSH_EVERY


class FrameBuffer:
    """Fixed-size FIFO cache mapping frame index → raw av.VideoFrame.

    Frames are stored in their native pixel format; conversion happens on
    retrieval, matching the existing behaviour of VideoHandler.

    Parameters
    ----------
    maxsize :
        Maximum number of frames to keep. When full the oldest-inserted
        entry is evicted before adding a new one.
    """

    def __init__(self, maxsize: int = 30) -> None:
        self._maxsize = maxsize
        self._cache: dict[int, av.VideoFrame] = {}
        self._order: deque[int] = deque()

    def get(self, idx: int) -> av.VideoFrame | None:
        """Return the cached frame for *idx*, or ``None`` on a miss."""
        return self._cache.get(idx)

    def put(self, idx: int, frame: av.VideoFrame) -> None:
        """Insert *frame* under *idx*, evicting the oldest entry if full."""
        if idx in self._cache:
            return
        if len(self._cache) == self._maxsize:
            evict = self._order.popleft()
            del self._cache[evict]
        self._cache[idx] = frame
        self._order.append(idx)

    def __contains__(self, idx: int) -> bool:
        return idx in self._cache


class VideoHandler(BaseAudioVideo):
    """
    Random-access video reader with timestamp-aware seeking.

    This class wraps a PyAV container to provide precise, timestamp-based
    random access to frames. It can return either `av.VideoFrame` objects
    or RGB float arrays in the shape ``(H, W, 3)`` normalized to ``[0, 1]``.
    Internally, a background thread builds an index of presentation timestamps
    (PTS) for fast seeks, with a fallback to on-the-fly indexing when the total
    frame count is unknown.

    Parameters
    ----------
    video_path :
        Path to the video file.
    stream_index :
        Index of the video stream to use, default is 0.
    time :
        Experimental timestamps for each frame (seconds). If ``None``, a
        uniform grid is generated from the stream's average rate.
    return_frame_array :
        If ``True`` (default), return frames as ``np.ndarray`` (RGB, float32 in ``[0, 1]``);
        otherwise return `av.VideoFrame` instances.

    Examples
    --------
    >>> from pynaviz.audiovideo import VideoHandler
    >>> vh = VideoHandler("example.mp4")  # doctest: +SKIP
    >>> # Get the frame at 1.5 seconds.
    >>> frame = vh.get(1.5)  # doctest: +SKIP
    >>> # Shape: (height, width, channels)
    >>> frame.shape  # doctest: +SKIP
    (480, 640, 3)
    >>> # Get frames from the second to the 10th, every other frame.
    >>> frame_sequence = vh[1:10:2]  # doctest: +SKIP
    >>> # Shape: (n_samples, height, width, channels)
    >>> frame_sequence.shape  # doctest: +SKIP
    (5, 480, 640, 3)
    """

    _thread_local = threading.local()

    def __init__(
        self,
        video_path: str | pathlib.Path,
        stream_index: int = 0,
        time: Optional[NDArray] = None,
        return_frame_array: bool = True,
        buffer_size: int = 30,
    ) -> None:
        super().__init__(video_path)
        self.stream = self.container.streams.video[stream_index]
        self.stream_index = stream_index
        self.return_frame_array = return_frame_array
        self._buffer = FrameBuffer(maxsize=buffer_size)
        # pts of the last frame *actually decoded* from the stream — used for
        # seek decisions.  current_frame can be updated by buffer / cache hits
        # without advancing the stream, so it must not be used for this purpose.
        self._stream_pts: int | None = None

        # default to linspace
        # TODO : what if number of frames is 0.
        if time is None:
            self._time_provided = False
            n_frames = self.stream.frames
            frame_duration = 1 / float(self.stream.average_rate)
            self.time = np.linspace(0, frame_duration * n_frames - frame_duration, n_frames)
        else:
            # TODO : check that number of time point matches number of frames
            self._time_provided = True
            self.time = np.asarray(time)

        # initialize index for last decoded frame
        # if sampling of other signals (LFP) is much denser, multiple times the frame
        # is unchanged, so cache the idx
        self.last_loaded_idx = None

        # initialize current frame
        self.current_frame: Optional[av.VideoFrame] = None

        if self.file_path.suffix == ".mkv":
            # mkv time is rounded to 3 digits, at least in the example video
            # generated by tests/generate_numbered_video.py
            self.round_fn = lambda x: np.round(x, 3)
        else:
            self.round_fn = lambda x: x

        # These will be initialized in the thread once n_frames is known
        self.all_pts: np.ndarray | list = []
        self.all_times = None
        self.key_mask = None

        self._i = 0  # number of committed (valid) PTS entries
        # None means the total frame count is not yet known (e.g. vp9/webm);
        # set to the final count by _build_index before signalling _index_ready.
        self._n_frames: int | None = self.stream.frames if self.stream.frames > 0 else None
        self._index_thread = threading.Thread(target=self._build_index, daemon=True)

        self._index_ready = threading.Event()
        self._index_thread.start()
        # decode first frame
        self.__getitem__(0)

    def reopen(self):
        """Reopen the video stream if it was previously closed. No-op if already open."""
        super().reopen()
        if self.stream is None:
            self.stream = self.container.streams.video[self.stream_index]
            self._stream_pts = None
            self._buffer = FrameBuffer(maxsize=self._buffer._maxsize)
            self.current_frame = None
            self.all_pts = []
            self.all_times = None
            self.key_mask = None
            self._i = 0
            self._n_frames = self.stream.frames if self.stream.frames > 0 else None
            self._index_ready = threading.Event()
            self._index_thread = threading.Thread(target=self._build_index, daemon=True)
            self._index_thread.start()
            self.__getitem__(0)

    @staticmethod
    def _ts_to_index(ts: float, time: NDArray) -> int:
        """
        Return the index of the frame whose experimental time is just before (or equal to) `ts`.

        Parameters
        ----------
        ts : float
            Experimental timestamp to match.
        time : NDArray
            Array of experimental timestamps, assumed sorted in ascending order,
            with one entry per frame.

        Returns
        -------
        idx : int
            Index of the frame with time <= `ts`. Clipped to [0, len(time) - 1].

        Notes
        -----
        - If `ts` is smaller than all values in `time`, returns 0.
        - If `ts` is greater than all values in `time`, returns `len(time) - 1`.
        """
        idx = np.searchsorted(time, ts, side="right") - 1
        return np.clip(idx, 0, len(time) - 1)

    def _extract_keyframe_times_and_points(
        self, video_path: str | pathlib.Path, stream_index: int = 0, first_only=False
    ) -> Tuple[NDArray, NDArray] | None:
        """
        Extract the indices and timestamps of keyframes from a video file.

        This function decodes the video while skipping non-keyframes, and records:
        - The index of each keyframe in the full video frame sequence
        - The "Presentation Time Stamp" to each keyframe.

        It is typically intended to run in a background thread during
        initialization of a ``VideoHandler``, and supports optimized seeking:

        - When the requested frame (based on experimental time) is before the
          current playback position, seeking backward is necessary.

        - When the requested frame is beyond the next known keyframe, seeking
          forward to the closest keyframe is more efficient than decoding all
          intermediate frames.

        Parameters
        ----------
        video_path : str or pathlib.Path
            The path to the video file.
        stream_index:
            The index of the video stream.
        first_only:
            If true, return the first keyframe only. Used at initialization.

        Returns
        -------
        keyframe_points : NDArray[float]
            The point number of the frame.

        keyframe_timestamps : NDArray[float]
            The timestamp of the frame.
        """
        keyframe_timestamp = []
        keyframe_pts = []

        with av.open(video_path) as container:
            stream = container.streams.video[stream_index]
            stream.codec_context.skip_frame = "NONKEY"

            frame_index = 0
            for frame in container.decode(stream):
                if not self._running:
                    return
                keyframe_timestamp.append(frame.time)
                keyframe_pts.append(frame.pts)
                if first_only:
                    break
                frame_index += 1

        return np.asarray(keyframe_pts), np.asarray(keyframe_timestamp, dtype=float)

    @contextmanager
    def _set_get_from_index(self, value):
        """Context manager for setting the shallow copy flag in a thread safe way."""
        # safe getattr is needed because the local variable is initialized
        # with every thread, and a thread won't have `get_from_index` since
        # in the main thread it is defined at __init__
        # which is not called by the thread.
        old_value = getattr(self._thread_local, "get_from_index", False)
        self._thread_local.get_from_index = value
        try:
            yield
        finally:
            self._thread_local.get_from_index = old_value

    def _extract_keyframes_pts(self):
        try:
            with av.open(self.file_path) as container:
                stream = container.streams.video[0]
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
            self._pts_keyframe_ready.set()

    def _build_index(self):
        try:
            with av.open(self.file_path) as container:
                stream = container.streams.video[self.stream_index]
                n_frames = stream.frames
                ctx = stream.codec_context
                has_b_frames = bool(ctx.has_b_frames)
                # guard against max_b_frames set to None for non-b-frame codecs
                max_b_frames = max(getattr(ctx, "max_b_frames", 1) or 1, 1)
                process = sorted if has_b_frames else lambda x: x
                temp = []
                # setup config for fixed-size and variable size index.
                if n_frames > 0:
                    # preallocate indices
                    with self._lock:
                        self.all_pts = np.empty(n_frames, dtype=np.int64)

                    def update(extracted_pts):
                        chunk = process(extracted_pts)
                        with self._lock:
                            self.all_pts[self._i: self._i + len(chunk)] = chunk
                            self._i += len(chunk)
                        extracted_pts.clear()
                else:
                    def update(extracted_pts):
                        chunk = process(extracted_pts)
                        with self._lock:
                            self.all_pts.extend(chunk)
                            self._i = len(self.all_pts)
                        extracted_pts.clear()

                # extraction loop: demux only — no decode — sort per GOP if needed.
                count_key_frames = 0
                for packet in container.demux(stream):
                    if not self._running:
                        return
                    if packet.pts is None or packet.pts < 0:
                        continue

                    count_key_frames += packet.is_keyframe
                    if _needs_flush(count_key_frames, temp, has_b_frames, max_b_frames):
                        update(temp)
                        count_key_frames = 0
                    temp.append(packet.pts)

                if temp:
                    update(temp)
                with self._lock:
                    self.all_pts = np.asarray(self.all_pts[: self._i], dtype=np.int64)

        except Exception as e:
            print("Index thread error:", e)
        finally:
            self._n_frames = self._i
            if self._time_provided and len(self.time) != self._i:
                warnings.warn(
                    f"The provided time array has length {len(self.time)}, but the video has {self._i} frames. "
                    "Overriding time with `np.linspace(time[0], time[-1], n_frames)`.",
                    UserWarning,
                    stacklevel=2,
                )
                self.time = np.linspace(self.time[0], self.time[-1], self._i)
            elif not self._time_provided and len(self.time) != self._i:
                frame_duration = 1 / float(self.stream.average_rate)
                self.time = np.linspace(0, frame_duration * self._i - frame_duration, self._i)
            self._index_ready.set()

    def _get_frame_idx(self, pts: int) -> int:
        """
        Get the frame index from the presentation time stamp.

        Parameters
        ----------
        pts:
            The presentation time stamp of the frame.

        Returns
        -------
        idx:
            The frame index corresponding to the given pts.
        use_time:
            If true, search using presentation time in seconds, otherwise use pts.

        """
        # Wait until enough index is available
        # Estimate pts from index (using filled index if available)
        with self._lock:
            done = self._i > 0 and self.all_pts[self._i - 1] > pts
        if done:
            # the pts for this timestamp has been filled
            idx = np.searchsorted(self.all_pts[: self._i], pts, side="left")
            use_time = False
        else:
            # keep going until at least two frames have been decoded by the thread
            while True:
                with self._lock:
                    if self._i > 1:
                        break
                time.sleep(0.001)
            # use recent history to get the step estimate
            with self._lock:
                # Linear extrapolation from available pts (use last 10 steps for an estimate)
                start, stop = max(self._i - 10, 0), self._i
                avg_step = np.mean(np.diff(self.all_pts[start:stop]))
                idx = int((pts - self.all_pts[0]) / avg_step)
                use_time = True
        return idx, use_time

    def _get_target_frame_pts(self, idx: int) -> Tuple[int, bool]:
        """
        Get the target frame presentation time stamp from frame index.

        Parameters
        ----------
        idx:
            The frame index.

        Returns
        -------
        target_pts:
            The target frame presentation time stamp corresponding to the frame index.
        use_time:
            If true, search using presentation time in seconds, otherwise use pts.

        """
        # Wait until enough index is available
        # Estimate pts from index (using filled index if available)
        with self._lock:
            done = self._i > idx
        if done:
            # the pts for this timestamp has been filled
            target_pts = self.all_pts[idx]
            use_time = False
        else:
            # keep going until at least two frames have been decoded by the thread
            while True:
                with self._lock:
                    if self._i > 1:
                        break
                time.sleep(0.001)
            # use recent history to get the step estimate
            with self._lock:
                # Linear extrapolation from available pts (use last 10 steps for an estimate)
                start, stop = max(self._i - 10, 0), self._i
                avg_step = np.mean(np.diff(self.all_pts[start:stop]))
                target_pts = int(self.all_pts[0] + avg_step * idx)
                use_time = True
        return target_pts, use_time

    def _get_key_frame(self, backward) -> av.VideoFrame | NDArray:
        idx = self.last_loaded_idx
        if idx is None:
            # fallback to safe keyframe
            self._wait_for_key_pts()
            if len(self._keyframe_pts) > 0:
                idx = self._get_frame_idx(self._keyframe_pts[0])[0] + 1
            else:
                idx = 0  # safe fallback

        # Get the pts of the last loaded index
        target_pts, use_time = self._get_target_frame_pts(idx)

        # Seek the next or previous keyframe based on the direction
        with self._lock:
            delta = max(np.mean(np.diff(self._keyframe_pts[:10])) // 2, 1)
        try:
            self.container.seek(
                int(
                    target_pts + (-delta if backward else delta)
                ),  # if you're on top of a key frame, seek does not move no matter what
                backward=backward,
                any_frame=False,
                stream=self.stream,
            )
        except av.error.PermissionError:
            # seek backward at the end of the file
            self.container.seek(
                int(target_pts),
                backward=True,
                any_frame=False,
                stream=self.stream,
            )

        # Decode the next frame, which should be a keyframe
        frame = next(
            frame
            for packet in self.container.demux(self.stream)
            if packet is not None
            for frame in packet.decode()
        )

        self.current_frame = frame

        # Get the index of the key frame
        self.last_loaded_idx = self._get_frame_idx(frame.pts)[0]

        # Return both
        return (
            self.current_frame.to_ndarray(format="rgb24")[::-1] / 255.0
            if self.return_frame_array
            else self.current_frame,
            self.last_loaded_idx,
        )

    def get(self, ts: float) -> av.VideoFrame | NDArray:
        """
        Return the frame at (or immediately preceding) a timestamp.

        Parameters
        ----------
        ts : float
            Target time in seconds.

        Returns
        -------
        :
            If ``return_frame_array`` is ``True``, returns an array with shape
            ``(H, W, 3)`` (RGB, float32 in ``[0, 1]``). Otherwise returns an
            `av.VideoFrame`.

        Notes
        -----
        - Seeks to the closest keyframe behind ``ts`` and decodes forward
          until the target is reached.
        - Uses an internal cache: if the requested frame index matches the
          previously decoded one, the cached frame is returned.
        """
        if not getattr(self._thread_local, "get_from_index", False):
            idx = self._ts_to_index(ts, self.time)
        else:
            idx = ts

        if idx == self.last_loaded_idx:
            return (
                self.current_frame.to_ndarray(format="rgb24")[::-1] / 255.0
                if self.return_frame_array
                else self.current_frame
            )

        cached = self._buffer.get(idx)
        if cached is not None:
            self.current_frame = cached
            self.last_loaded_idx = idx
            return (
                cached.to_ndarray(format="rgb24")[::-1] / 255.0
                if self.return_frame_array
                else cached
            )

        target_pts, use_time = self._get_target_frame_pts(idx)

        if self._stream_pts is None or self._need_seek_call(self._stream_pts, target_pts):
            self.container.seek(
                int(target_pts), backward=True, any_frame=False, stream=self.stream
            )

        # Decode forward from the keyframe until the frame just before (or equal to) target_pts
        last_idx, preceding_frame = self._decode_and_check_frames(use_time, target_pts, idx)

        if preceding_frame is not None:
            self.last_loaded_idx = idx
            self.current_frame = preceding_frame
            self._stream_pts = preceding_frame.pts
            self._buffer.put(idx, preceding_frame)

        return (
            self.current_frame.to_ndarray(format="rgb24")[::-1] / 255.0
            if self.return_frame_array
            else self.current_frame
        )

    def _frame_iterator(self, fall_back_pts: int | None):
        """
        Safe frame iterator.

        Iterate frames from current stream location. If End-of-File error is
        hit, seek to pts and iterate over frames from there.
        """
        try:
            for packet in self.container.demux(self.stream):
                if packet is None:
                    continue
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    yield frame
        except av.error.EOFError as e:
            if fall_back_pts is None:
                raise e
            self.container.seek(
                int(fall_back_pts), backward=True, any_frame=False, stream=self.stream
            )
            yield from self._frame_iterator(None)

    def _decode_and_check_frames(self, use_time: bool, target_pts: int, idx: int):
        """Decode from stream."""
        preceding_frame = None
        last_idx = self.last_loaded_idx
        frame_duration = 1 / float(self.stream.average_rate)
        time_threshold = self.round_fn(idx * frame_duration)

        for frame in self._frame_iterator(target_pts):
            if frame.pts is None:
                continue
            if (not use_time and frame.pts > target_pts) or (
                use_time and frame.time > time_threshold
            ):
                last_idx = idx
                current_frame = preceding_frame or frame
                return last_idx, current_frame
            elif (not use_time and frame.pts == target_pts) or (
                use_time and frame.time == time_threshold
            ):
                last_idx = idx
                current_frame = frame
                return last_idx, current_frame
            preceding_frame = frame
        return last_idx, preceding_frame

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        :
            Shape of the video, ``(n_frames, width, height)``.

        Notes
        -----
        - When the total frame count is unknown at initialization, the length
          may grow while the background indexer discovers frames. A warning is
          emitted until indexing is complete.
        """
        if self._n_frames is None:
            self._wait_for_all_pts()
        return self._n_frames, self.stream.width, self.stream.height

    @property
    def index(self) -> NDArray:
        """
        Time index in seconds corresponding to frames.

        If ``time`` was provided at initialization, that array is returned.
        Otherwise, a uniformly spaced array derived from the stream rate is used
        and may be updated as indexing progresses.
        """
        return self.time

    @property
    def t(self) -> NDArray:
        """
        Time index in seconds corresponding to frames.

        If ``time`` was provided at initialization, that array is returned.
        Otherwise, a uniformly spaced array derived from the stream rate is used
        and may be updated as indexing progresses.
        """
        return self.time

    def _wait_for_all_pts(self, timeout=None):
        """Wait until the PTS index thread has finished."""
        self._index_ready.wait(timeout)

    def _wait_for_key_pts(self, timeout=None):
        """Wait until the keyframe PTS thread has finished."""
        self._pts_keyframe_ready.wait(timeout)

    def _wait_for_index(self, timeout=None):
        """Wait until both the PTS index and keyframe threads have finished."""
        self._wait_for_all_pts(timeout)
        self._wait_for_key_pts(timeout)

    def get_slice(self, start: float, end: float = None):
        # TODO check start and end are sorted
        start = self._ts_to_index(start, self.time)
        if end:
            end = self._ts_to_index(end, self.time)
            return slice(start, end)
        else:
            return slice(start, start + 1)

    def _append_frame(self, frames, idx, frame):
        if self.return_frame_array:
            frames[idx] = frame.to_ndarray(format="rgb24")[::-1] / 255.0
        else:
            frames.append(frame)

    def _decode_multiple(
        self,
        idx_start: int,
        idx_end: int,
        step: int = 1,
    ) -> Tuple[int, List[av.VideoFrame] | NDArray, av.VideoFrame]:
        n_frames = self._n_frames if self._n_frames is not None else self.shape[0]
        effective_end = min(idx_end, n_frames)
        indices = np.arange(idx_start, effective_end, step)
        num_frames = len(indices)
        time_threshold_all = self.round_fn(indices)

        if self.return_frame_array:
            frames = np.empty(
                (num_frames, self.stream.height, self.stream.width, 3),
                dtype=np.float32,
            )
        else:
            frames = []

        collected = 0

        # initialize current frame
        if self.current_frame is None:
            self.get(0)

        preceding_frame = self.current_frame
        last_frame = self.current_frame
        decoder = None  # frame-level iterator; reset after every seek

        while collected < num_frames:
            # check buffer first
            cached = self._buffer.get(indices[collected])
            if cached is not None:
                self.current_frame = cached
                self.last_loaded_idx = indices[collected]
                self._append_frame(frames, collected, cached)
                preceding_frame = cached
                last_frame = cached
                collected += 1
                continue

            target_pts, use_time = self._get_target_frame_pts(indices[collected])

            # Open a decoder (or re-open after a seek) when needed.
            if decoder is None or (
                self._need_seek_call(self._stream_pts, target_pts)
            ):
                self.container.seek(
                    int(target_pts), backward=True, any_frame=False, stream=self.stream
                )
                decoder = self.container.decode(self.stream)

            # Advance one frame. container.decode handles B-frame buffering
            # internally, so zero-frame packets are transparent to us.
            try:
                frame = next(f for f in decoder if f.pts is not None)
                self._buffer.put(self._get_frame_idx(frame.pts)[0], frame)
                self._stream_pts = frame.pts
            except StopIteration:
                break

            time_threshold = time_threshold_all[collected]
            found_next = (
                (frame.pts > target_pts) if not use_time else (frame.time > time_threshold)
            )
            found_current = (
                (frame.pts == target_pts) if not use_time else (frame.time == time_threshold)
            )

            if found_next:
                frame = preceding_frame or frame
                self._append_frame(frames, collected, frame)
                collected += 1
            elif found_current:
                self._append_frame(frames, collected, frame)
                collected += 1

            last_frame = frame
            preceding_frame = frame

        return indices[-1], frames, last_frame

    def __getitem__(self, idx: slice | int) -> NDArray | av.VideoFrame | List[av.VideoFrame]:
        """
        Get item for video frame.

        Gets one or more frames from a video.

        Parameters
        ----------
        idx:
            The index for slicing, can be a slice or a integer.

        Returns
        -------
        ndarray or av.VideoFrame or list[av.VideoFrame]
            - If indexing with an ``int``:
              returns a single frame (array or `av.VideoFrame`).
            - If indexing with a ``slice``:
              returns a stack of frames as ``ndarray`` with shape
              ``(n_frames, height, width, 3)`` when ``return_frame_array`` is ``True``,
              otherwise a ``list[av.VideoFrame]``.
        """
        if isinstance(idx, slice):
            # Resolve frame count once — fast path if already known, otherwise waits.
            n_frames = self._n_frames if self._n_frames is not None else self.shape[0]

            # Fill in missing slice components
            start = idx.start or 0
            if start >= n_frames:
                if self.return_frame_array:
                    return np.empty((0, self.stream.height, self.stream.width, 3))
                else:
                    return []
            stop = idx.stop if idx.stop is not None else n_frames
            step = idx.step if idx.step is not None else 1

            # convert negative vals
            start = start if start >= 0 else start + n_frames
            start = max(0, min(start, n_frames))
            stop = stop + n_frames if stop < 0 else stop
            stop = max(0, min(stop, n_frames))

            # revert slice if negative step
            revert = step < 0
            step = abs(step)

            if (stop - start) // step > 1:
                target_pts, use_time = self._get_target_frame_pts(start)

                if self._stream_pts is None or self._need_seek_call(
                    self._stream_pts, target_pts
                ):
                    self.container.seek(
                        int(target_pts), backward=True, any_frame=False, stream=self.stream
                    )

                frame_idx, frames, last_frame = self._decode_multiple(
                    start, stop, step=step
                )
                # update current decoded frame
                if len(frames):
                    self.last_loaded_idx = frame_idx
                    self.current_frame = last_frame
                    self._stream_pts = last_frame.pts
                return frames if not revert else frames[::-1]

        # Default case: single index
        with self._set_get_from_index(True):
            # TODO Check borders
            idx_start = idx if not hasattr(idx, "start") else idx.start
            n_frames = self._n_frames if self._n_frames is not None else self.shape[0]
            idx_start = idx_start if idx_start >= 0 else n_frames + idx_start
            frame = self.get(idx_start)
            # handle slice requesting a single frame:
            # for arrays add 1 dimension (1, pixel, pixel)
            # for frames return a len 1 list.
            if isinstance(idx, slice):
                if isinstance(frame, np.ndarray):
                    frame = np.expand_dims(frame, axis=0)
                else:
                    frame = [frame]

        return frame

    def __len__(self):
        return self.shape[0]
