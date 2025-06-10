import pytest
import av
import numpy as np
from pynaviz import video_handling


@pytest.fixture()
def video_info(request):
    extension = request.param
    video = f"test_video/numbered_video.{extension}"

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
    handler = video_handling.VideoHandler(video, time=np.arange(100))
    frame = handler.get(requested_frame_ts)
    expected_pts = frame_pts_ref[expected_frame_id]
    assert frame.pts == expected_pts

