import pathlib

import av
import imageio.v3 as iio
import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest

from pynaviz import PlotVideo
from pynaviz.audiovideo import video_handling
from pynaviz.utils import GRADED_COLOR_LIST


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
def test_pts_ordering(video_info):
    frame_pts, _, video = video_info
    with video_handling.VideoHandler(video, time=np.arange(100)) as handler:
        handler._wait_for_index(15)
        np.testing.assert_array_equal(handler.all_pts, np.sort(handler.all_pts))
        np.testing.assert_array_equal(
            handler.all_pts, np.asarray(frame_pts, dtype=handler.all_pts.dtype)
        )


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


@pytest.mark.parametrize("tsdframe_shape", [(10, 2), (10, 4), (10, 6)])
@pytest.mark.parametrize("video_info", ["mp4"], indirect=True)
def test_points_overlay(video_info, tsdframe_shape: tuple[int, int]):
    _, _, video = video_info
    video = pathlib.Path(video)
    video_obj = PlotVideo(video, t=np.arange(100), start_worker=False)
    tsdframe = nap.TsdFrame(
        t=np.arange(tsdframe_shape[0]),
        d=100 + np.random.randn(*tsdframe_shape)
    )
    video_obj.superpose_points(tsdframe, color = "red", markersize=3, thickness=5, label="label")
    state = video_obj.points["label"].get_state()
    if tsdframe_shape[1] > 2:
        assert state == {"color": (1., 0., 0., 1.), "markersize": 3, "thickness": 5}
    else:
        assert state == {"color": (1., 0., 0., 1.), "markersize": 3}
    video_obj.close()


@pytest.mark.parametrize("tsdframe_shape", [(10, 4)])
@pytest.mark.parametrize("video_info", ["mp4"], indirect=True)
@pytest.mark.parametrize("color", ["red", None])
@pytest.mark.parametrize("label", ["label", None])
def test_points_overlay_params(video_info, tsdframe_shape: tuple[int, int], color, label):
    _, _, video = video_info
    video = pathlib.Path(video)
    video_obj = PlotVideo(video, t=np.arange(100), start_worker=False)
    tsdframe = nap.TsdFrame(
        t=np.arange(tsdframe_shape[0]),
        d=100 + np.random.randn(*tsdframe_shape)
    )
    video_obj.superpose_points(tsdframe, color = color, markersize=3, thickness=5, label=label)
    if label is None:
        actual_label = "points_1"
    else:
        actual_label = label
    if color is None:
        color = gfx.Color(GRADED_COLOR_LIST[0]).rgba
    else:
        color = gfx.Color(color).rgba
    state = video_obj.points[actual_label].get_state()
    assert state == {"color": color, "markersize": 3., "thickness": 5.}
    video_obj.close()

@pytest.mark.parametrize("video_info", ["mp4"], indirect=True)
def test_roundtrip_points(video_info):
    _, _, video = video_info
    tsdframe = nap.TsdFrame(
        t=np.arange(10),
        d=100 + np.random.randn(10, 4)
    )
    video_obj = PlotVideo(video, t=np.arange(100), start_worker=False)
    video_obj.superpose_points(tsdframe, color=None, markersize=3, thickness=5, label="label")
    video_obj.renderer.render(video_obj.scene, video_obj.camera)
    img = video_obj.renderer.snapshot()
    state = video_obj.get_plot_state()
    video_obj.close()
    video_obj = PlotVideo(video, t=np.arange(100), start_worker=False)
    video_obj.set_plot_state(state, available_vars={"label": tsdframe})
    video_obj.renderer.render(video_obj.scene, video_obj.camera)
    img2 = video_obj.renderer.snapshot()
    np.testing.assert_allclose(img, img2, atol=1)
    video_obj.close()


# ---------------------------------------------------------------------------
# Buffer integration tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_get_populates_buffer(video_info):
    """A frame decoded via get() should land in the buffer."""
    _, _, video_path = video_info
    with video_handling.VideoHandler(
        video_path, time=np.arange(100), return_frame_array=False
    ) as video:
        video.get(5.0)
        idx = video_handling.VideoHandler._ts_to_index(5.0, video.time)
        assert idx in video._buffer


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_get_hits_buffer_on_second_call(video_info):
    """Second get() for the same timestamp must return the cached frame
    without re-decoding — proved by closing the container between calls."""
    _, _, video_path = video_info
    with video_handling.VideoHandler(
        video_path, time=np.arange(100), return_frame_array=False
    ) as video:
        first = video.get(5.0)

        # Close the underlying container so any attempt to decode would raise.
        video.container.close()
        video.container = None

        second = video.get(5.0)

    assert first.pts == second.pts


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_getitem_int_populates_buffer(video_info):
    """video[idx] (integer index) routes through get(), so the frame should
    end up in the buffer."""
    _, _, video_path = video_info
    with video_handling.VideoHandler(
        video_path, time=np.arange(100), return_frame_array=False
    ) as video:
        idx = 10
        video[idx]
        assert idx in video._buffer


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_getitem_int_hits_buffer_on_second_call(video_info):
    """video[idx] called twice: second call must come from the buffer."""
    _, _, video_path = video_info
    with video_handling.VideoHandler(
        video_path, time=np.arange(100), return_frame_array=False
    ) as video:
        idx = 10
        first = video[idx]

        video.container.close()
        video.container = None

        second = video[idx]

    assert first.pts == second.pts


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_buffer_fifo_eviction_in_handler(video_info):
    """Accessing more than buffer_size distinct frames evicts the earliest one."""
    _, _, video_path = video_info
    buf_size = 5
    with video_handling.VideoHandler(
        video_path, time=np.arange(100), return_frame_array=False, buffer_size=buf_size
    ) as video:
        # warm up: access buf_size + 1 distinct frames (skip 0; __init__ decodes it)
        for i in range(1, buf_size + 2):
            video[i]

        # frame at index 1 was inserted first and should have been evicted
        assert 1 not in video._buffer
        # the most recent buf_size frames must still be present
        for i in range(2, buf_size + 2):
            assert i in video._buffer


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_getitem_decoupled_current_frame_and_pts_decoding_via_get(video_info):
    _, _, video_path = video_info
    with video_handling.VideoHandler(
        video_path, time=np.arange(100), return_frame_array=False
    ) as video_obj:
        # populate the buffer with frames 10 and 11
        _ = video_obj[10]
        _ = video_obj[11]
        # move current_frame back to 10 via buffer (should not advance the stream)
        _ = video_obj[10]
        # current_frame and stream pts must now differ
        assert video_obj.current_frame.pts != video_obj._stream_pts
        # decoding the next frame must still be correct
        frame = video_obj[12]
        with video_obj._set_get_from_index(True):
            video_obj.get(12)
            assert video_obj.current_frame.pts == frame.pts


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
def test_getitem_decoupled_current_frame_and_pts_decoding_via_decode_multiple(video_info):
    _, _, video_path = video_info
    with video_handling.VideoHandler(
        video_path, time=np.arange(100), return_frame_array=False
    ) as video_obj:
        # populate the buffer via a slice
        _ = video_obj[10:12]
        # move current_frame back to 10 via buffer (should not advance the stream)
        _ = video_obj[10]
        # current_frame and stream pts must now differ
        assert video_obj.current_frame.pts != video_obj._stream_pts
        # decoding the next slice must still be correct
        frames = video_obj[12:14]
        with video_obj._set_get_from_index(True):
            for i, frame in zip(range(12, 14), frames):
                video_obj.get(i)
                assert video_obj.current_frame.pts == frame.pts
