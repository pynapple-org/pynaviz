import pathlib

import av
import imageio.v3 as iio
import numpy as np
import pytest

from pynaviz import PlotVideo
from pynaviz.audiovideo import video_handling


@pytest.fixture()
def video_info(request):
    extension = request.param
    video = pathlib.Path(__file__).parent / f"test_video/numbered_video.{extension}"

    frame_pts = []
    keyframe_pts = []

    with av.open(video) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONE"  # decode all frames

        for frame in container.decode(stream):
            frame_pts.append(frame.pts)
            if frame.key_frame:
                keyframe_pts.append(frame.pts)

    return frame_pts, keyframe_pts, video


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
@pytest.mark.parametrize(
    "requested_frame_ts, expected_frame_id",
    [(0, 0), (0.1, 0), (1.0, 1), (1.1, 1), (1.6, 1), (99, 99), (99.6, 99), (111, 99)],
)
def test_video_handler(video_info, requested_frame_ts, expected_frame_id):
    frame_pts_ref, _, video = video_info
    with video_handling.VideoHandler(
            video, time=np.arange(100), return_frame_array=False
        ) as handler:
        frame = handler.get(requested_frame_ts)
        expected_pts = frame_pts_ref[expected_frame_id]
        assert frame.pts == expected_pts


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
@pytest.mark.parametrize(
    "requested_frame_ts, expected_frame_id",
    [(0, 0), (0.1, 0), (1.0, 1), (1.1, 1), (1.6, 2), (99, 99), (99.6, 99), (111, 99)],
)
def test_video_handler_get_frame_snapshots(
    video_info, requested_frame_ts, expected_frame_id
):
    _, _, video = video_info
    extension = pathlib.Path(video).suffix[1:]
    path = (
        pathlib.Path(__file__).parent
        / f"screenshots/video/numbered_video_{extension}_frame_{expected_frame_id}.png"
    )
    stored_img = iio.imread(path)
    print("\nread image")
    v = PlotVideo(video, t=np.arange(100), start_worker=False)
    print("opened plot video")
    v.set_frame(requested_frame_ts)
    v.renderer.render(v.scene, v.camera)
    img = v.renderer.snapshot()
    # tolerance equal to this pygfx example test
    # https://github.com/pygfx/pygfx/blob/main/examples/tests/test_examples.py#L116
    atol = 1
    np.testing.assert_allclose(img, stored_img, atol=atol)
    v.close()
    v = None


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_getitem_single_index_return_frame(video_info):
    _, _, video_path = video_info
    with video_handling.VideoHandler(
            video_path, time=np.arange(100), return_frame_array=False
        ) as video:
        idx = 7
        frame = video[idx]
        assert isinstance(frame, av.VideoFrame), "Single frame should be a single frame"


@pytest.mark.parametrize(
    "start, stop, step",
    [
        (0, 5, 1),  # simple forward slice
        (10, 20, 2),  # skip frames
        (25, None, 3),  # slice to end with step
        (None, 10, 1),  # from beginning
        (None, None, 5),  # full range with step
    ],
)
@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_getitem_slice_return_frame(video_info, start, stop, step):
    _, _, video_path = video_info
    with video_handling.VideoHandler(
            video_path, time=np.arange(100), return_frame_array=True
        ) as video:
        idx = slice(start, stop, step)
        frames = video[idx]

        # Compute expected length
        start_idx = start or 0
        stop_idx = stop if stop is not None else video.shape[0]
        step_val = step or 1
        expected_len = len(range(start_idx, stop_idx, step_val))

        assert isinstance(frames, np.ndarray), "Sliced result should be a list of frames"
        assert np.isdtype(np.float32, frames.dtype)
        assert (
            len(frames) == expected_len
        ), f"Expected {expected_len} frames but got {len(frames)}"
        assert all(fi.shape == (video.shape[2], video.shape[1], 3) for fi in frames)


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_getitem_single_index_return_frame2(video_info):
    _, _, video_path = video_info
    with video_handling.VideoHandler(
            video_path, time=np.arange(100), return_frame_array=True
        ) as video:
        idx = 7
        frame = video[idx]
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (video.shape[2], video.shape[1], 3)


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
@pytest.mark.parametrize(
    "start, stop, step",
    [
        (0, 5, 1),
        (10, 20, 2),
        (95, 100, 1),
        (99, 100, 1),
        (0, 100, 25),
    ],
)
def test_getitem_slice_matches_expected(video_info, start, stop, step):
    _, _, video = video_info
    video = pathlib.Path(video)
    video_obj = PlotVideo(video, t=np.arange(100), start_worker=False)
    video_obj.data.return_frame_array = False
    frames = video_obj.data[start:stop:step]
    video_obj.data.return_frame_array = True
    # make sure the video meta-info about time are fully computed
    video_obj.data._wait_for_index(timeout=15)
    for i, frame in zip(range(start, stop, step), frames):
        with video_obj.data._set_get_from_index(True):
            video_obj.set_frame(video_obj.data.time[i])
            test_frame = video_obj.data.current_frame
            assert test_frame.pts == frame.pts, "Frame pts mismatch."
            # check if the decoding was correct
            # (assuming current frame is decoded correctly, which is tested above in
            # test_video_handler_get_frame_snapshots)
            np.testing.assert_array_equal(
                video_obj.data.current_frame.to_ndarray(), frame.to_ndarray()
            )
    video_obj.close()


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_getitem_multiple_times(video_info):
    _, _, video = video_info
    video = pathlib.Path(video)
    video_obj = PlotVideo(video, t=np.arange(100), start_worker=False)
    frames = video_obj.data[1:12:2]
    frames2 = video_obj.data[1:12:2]
    np.testing.assert_array_equal(frames, frames2)
    video_obj.close()


# @pytest.mark.parametrize(
#     "start, stop, step",
#     [
#         (0, 5, 1),
#         (10, 20, 2),
#         (95, 100, 1),
#         (99, 100, 1),
#         (0, 100, 25),
#     ],
# )
# @pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
# def test_getitem_negative_slicing_step(video_info):
