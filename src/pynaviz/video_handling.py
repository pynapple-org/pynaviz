import pathlib

from typing import Optional, Tuple
import av
import numpy as np
from numpy.typing import NDArray
import threading


def extract_keyframe_indices_and_points(
        video_path: str | pathlib.Path, first_only=False
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
    first_only:
        If true, return the first keypoint only. Used at initialization.

    Returns
    -------
    keyframe_indices : NDArray[int]
        The indices of keyframes in the full decoded video stream.

    keyframe_points : NDArray[float]
        The point number of the frame.

    """
    keyframe_indices = []
    keyframe_pts = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"

        frame_index = 0  # absolute frame index in the full stream
        for frame in container.decode(stream):
            keyframe_indices.append(frame_index)
            keyframe_pts.append(frame.pts)
            if first_only:
                break
            frame_index += 1

    return np.asarray(keyframe_indices, dtype=int), np.asarray(keyframe_pts)


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
        self.video_path = video_path
        self.container = av.open(video_path)
        self.stream = self.container.streams.video[stream_index]

        # default to linspace
        if time is None:
            n_frames = self.stream.frames
            frame_duration = 1 / float(self.stream.average_rate)
            self.time = np.linspace(0, frame_duration * n_frames, n_frames)
        else:
            self.time = np.asarray(time)

        # initialize keyframe ts to empty array
        self.keyframe_pts, self.keyframe_indices = extract_keyframe_indices_and_points(video_path, True)

        # run a thread extracting the keyframe
        self.keyframe_lock = threading.Lock()
        threading.Timer(0.0001, self.extract_keyframe_indices_and_timestamps).start()

        # initialize decoded frame last index
        self.last_idx = None

        # initialize current frame
        self.current_frame = None
        self.current_packet_list = []


    def extract_keyframe_indices_and_timestamps(self):
        keyframe_indices, keyframes = extract_keyframe_indices_and_points(self.video_path)
        with self.keyframe_lock:
            self.keyframe_pts = keyframes
            self.keyframe_indices = keyframe_indices


    def seek(self, pts_keyframe):
        """Find the nearest keypoint frame``.

        This function navigates to the nearest keypoint frame before ``ts`` and
        defines an iterator for streaming from the keypoint onwards.
        Each frame must be decoded in sequence from the keypoint.
        """
        # calculate the number of points
        self.container.seek(int(pts_keyframe), backward=True, any_frame=False, stream=self.stream)
        self.packet_iter = self.container.demux(self.stream)


    def get(self, ts: float):
        idx = ts_to_index(ts, self.time)

        # if ts did not change frame, return current
        if idx == self.last_idx:
            return self.current_frame

        # if the frame  has changed compute:
        # - the previous keypoint
        # - the keypoint following the last visited frame
        with self.keyframe_lock:
            previous_keypoint_idx = np.clip(
                np.searchsorted(self.keyframe_pts, idx, side="right") - 1,
                0,
                len(self.keyframe_pts) - 1
            )
            if self.last_idx:
                next_keypoint_idx = np.searchsorted(self.keyframe_pts, self.last_idx, side="right")
            else:
                # make sure that the next keypoint idx is larger than the current idx
                # if it's the first frame visited
                next_keypoint_idx = self.stream.frames + 1


        # seek (which re-initialize packet_iter) if
        # - it's the fist get called (self.last_idx is None)
        # - the video has been scrolled backwards (the current index precedes last index)
        # - one scrolled ahead of more than one keypoint
        need_seek = (
                self.last_idx is None or idx < self.last_idx or next_keypoint_idx <= previous_keypoint_idx
        )

        if need_seek:
            with self.keyframe_lock:
                self.seek(self.keyframe_pts[previous_keypoint_idx])
                self.current_packet_list = next(self.packet_iter).decode()

        # compute the number of frame to decode
        n_frames_iter = idx - (previous_keypoint_idx if need_seek else self.last_idx)

        # if current packet include frame, return frame and crop current_packet_list
        if n_frames_iter < len(self.current_packet_list):
            self.current_frame = self.current_packet_list[n_frames_iter]
            self.last_idx = idx
            self.current_packet_list = self.current_packet_list[n_frames_iter + 1:]
            return self.current_frame.to_ndarray(format="yuv420p")

        # if not, iterate over the following packets
        frame_counter = 0 if need_seek else len(self.current_packet_list)
        for packet in self.packet_iter:
            self.current_packet_list = packet.decode()
            frame_in_packet_counter = 0
            for frame in self.current_packet_list:
                if frame_counter == n_frames_iter - 1:
                    self.current_frame = frame
                    self.last_idx = idx
                    self.current_packet_list = self.current_packet_list[frame_in_packet_counter+1:]
                    return self.current_frame.to_ndarray(format="yuv420p")
                frame_in_packet_counter += 1
                frame_counter += 1
