import pathlib

import pytest
import av
import numpy as np
from pynaviz import PlotVideo, video_handling
import imageio.v3 as iio

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
@pytest.mark.parametrize("requested_frame_ts, expected_frame_id", [(0, 0), (0.1, 0), (1., 1), (1.1, 1), (1.6, 1), (99, 99), (99.6, 99), (111, 99)])
def test_video_handler(video_info, requested_frame_ts, expected_frame_id):
    frame_pts_ref, _, video = video_info
    handler = video_handling.VideoHandler(video, time=np.arange(100), return_frame_array=False)
    frame = handler.get(requested_frame_ts)
    expected_pts = frame_pts_ref[expected_frame_id]
    assert frame.pts == expected_pts


@pytest.mark.parametrize("video_info", ["mp4", "mkv", "avi"], indirect=True)
@pytest.mark.parametrize("requested_frame_ts, expected_frame_id", [(0, 0), (0.1, 0), (1., 1), (1.1, 1), (1.6, 2), (99, 99), (99.6, 99), (111, 99)])
def test_video_handler_get_frame_snapshots(video_info, requested_frame_ts, expected_frame_id):
    _, _, video = video_info
    extension = pathlib.Path(video).suffix[1:]
    path = pathlib.Path(__file__).parent / f"screenshots/numbered_video_{extension}_frame_{expected_frame_id}.png"
    stored_img = iio.imread(path)
    v = PlotVideo(video, time=np.arange(100))
    v.set_frame(requested_frame_ts)
    v.renderer.render(v.scene, v.camera)
    img = v.renderer.snapshot()
    # tolerance equal to this pygfx example test
    # https://github.com/pygfx/pygfx/blob/main/examples/tests/test_examples.py#L116
    atol = 1
    np.testing.assert_allclose(img, stored_img, atol=atol)
