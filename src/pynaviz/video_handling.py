import pathlib
import threading
from typing import Optional, Tuple

import av
import numpy as np
from numpy.typing import NDArray


def extract_keyframe_times_and_points(
        video_path: str | pathlib.Path, stream_index: int = 0, first_only=False
) -> Tuple[NDArray, NDArray]:
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
        If true, return the first keypoint only. Used at initialization.

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
            keyframe_timestamp.append(frame.time)
            keyframe_pts.append(frame.pts)
            if first_only:
                break
            frame_index += 1

    return np.asarray(keyframe_pts), np.asarray(keyframe_timestamp,dtype=float)


def ts_to_index(ts: float, time: NDArray) -> int:
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
    idx = np.searchsorted(time, ts, side='right') - 1
    return np.clip(idx, 0, len(time) - 1)


class VideoHandler:
    """Class for getting video frames."""
    def __init__(self, video_path: str | pathlib.Path, stream_index=0,  time: Optional[NDArray]=None) -> None:
        self.video_path = pathlib.Path(video_path)
        self.container = av.open(video_path)
        self.stream = self.container.streams.video[stream_index]

        # default to linspace
        if time is None:
            n_frames = self.stream.frames
            frame_duration = 1 / float(self.stream.average_rate)
            self.time = np.linspace(0, frame_duration * n_frames - frame_duration, n_frames)
        else:
            self.time = np.asarray(time)

        # initialize keyframe ts decoding one packet only
        # the rest done in thread
        self.keyframe_pts, self.keyframe_timestamp  = extract_keyframe_times_and_points(video_path, stream_index, True)

        # run a thread extracting the keyframe
        self.keyframe_lock = threading.Lock()
        threading.Timer(0.0001, self.extract_keyframe_indices_and_timestamps).start()

        # initialize decoded frame last index
        # if sampling of other signals (LFP) is much denser, multiple times the frame
        # is unchanged, so cache the idx
        self.last_idx = None

        # initialize current frame
        self.current_frame = None

        if self.video_path.suffix == ".mkv":
            self.round_fn = lambda x: np.round(x, 3)
        else:
            self.round_fn = lambda x: x


    def extract_keyframe_indices_and_timestamps(self):
        """
        Loop over keypoints and extract their frame time stamp and presentation time stamp.
        """
        keyframes, keyframe_timestamp = extract_keyframe_times_and_points(self.video_path)
        with self.keyframe_lock:
            self.keyframe_pts = keyframes
            self.keyframe_timestamp = keyframe_timestamp


    def seek(self, pts_keyframe):
        """Find the nearest keypoint frame``.

        This function navigates to the nearest keypoint frame ``pts_keyframe`` and
        defines an iterator for streaming from the keypoint onwards.
        Each frame must be decoded in sequence from the keypoint.
        """
        # calculate the number of points
        self.container.seek(int(pts_keyframe), backward=True, any_frame=False, stream=self.stream)
        self.packet_iter = self.container.demux(self.stream)


    def get(self, ts: float):
        """Get the frame preceding ts.

        Parameters
        ----------
        ts : float
            Timestamp of the frame.
        """
        idx = ts_to_index(ts, self.time)

        # if ts did not change frame, return current
        if idx == self.last_idx:
            return self.current_frame

        # current duration from ts
        # mkv rounds to 3 decimals, so apply here too otherwise
        # search in mode ``closest`` may fail due to rounding errors.
        delta_t = self.round_fn(float(idx / self.stream.base_rate))

        # calculate putative keypoint
        with self.keyframe_lock:
            previous_keypoint_idx = np.clip(
                np.searchsorted(self.keyframe_timestamp, delta_t, side="right") - 1,
                0,
                len(self.keyframe_pts) - 1
            )
            self.seek(self.keyframe_pts[previous_keypoint_idx])

        # Decode frames from keyframe forward until delta_t is exceeded
        preceding_frame = None

        for packet in self.packet_iter:
            for frame in packet.decode():
                if frame.time is None:
                    continue  # Skip bad frames

                if frame.time > delta_t:
                    # Found a frame after target — return the preceding one
                    self.last_idx = idx
                    self.current_frame = preceding_frame or frame
                    return self.current_frame

                preceding_frame = frame

        # Fallback: return last decoded frame (end of video)
        self.last_idx = idx
        self.current_frame = preceding_frame
        return preceding_frame
