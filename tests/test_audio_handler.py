import pathlib
from typing import List, Tuple

import av
import numpy as np
import pytest
from numpy.typing import NDArray
from pynaviz.audiovideo import audio_handling

@pytest.fixture(scope="module")
def fully_decoded_audio(request) -> Tuple[pathlib.Path, List[NDArray], List[int], List[av.AudioFrame], List[int]]:
    extension = request.param
    audio = pathlib.Path(__file__).parent / f"test_audio/sine_audio.{extension}"
    frame_arrays: List[NDArray] = []
    frame_pts: List[int] = []
    frame_av: List[av.AudioFrame] = []
    frame_size: List[int] = []
    with av.open(audio) as container:
        stream = container.streams.audio[0]
        for packet in container.demux(stream):
            for frame in packet.decode():
                # this is to convince the mypy/pyright that
                # frame is of AudioFrame type
                frame: av.AudioFrame = frame
                if frame.pts is not None:
                    frame_av.append(frame)
                    fr = frame.to_ndarray()
                    frame_size.append(fr.shape[1])
                    frame_arrays.append(fr)
                    frame_pts.append(frame.pts)
    return audio, frame_arrays, frame_pts, frame_av, frame_size


@pytest.mark.parametrize("fully_decoded_audio", ["wav", "mp3", "flac"], indirect=True)
def test_av_handler_full_decoding(fully_decoded_audio):
    audio_path, frame_arrays, frame_pts, frame_av, frame_size = fully_decoded_audio
    with audio_handling.AudioHandler(audio_path) as handler:
        array = handler.get(0, handler.tot_length)
        concat_array = np.concatenate(frame_arrays, axis=1).T
        assert handler.shape == (88200, 1), "the video shape doesn't match expectation."
        assert handler.tot_length == 2, "the video duration in sec doesn't match expectation."
        np.testing.assert_array_equal(array, concat_array)
        # this may fail if the seek behavior is wrong
        array = handler.get(0, handler.tot_length)
        np.testing.assert_array_equal(array, concat_array)


@pytest.mark.parametrize("fully_decoded_audio", ["wav", "mp3", "flac"], indirect=True)
def test_av_handler_partial_decoding(fully_decoded_audio):
    audio_path, frame_arrays, frame_pts, frame_av, frame_size = fully_decoded_audio
    with audio_handling.AudioHandler(audio_path) as handler:
        array = handler.get(0, handler.tot_length)
        concat_array = np.concatenate(frame_arrays, axis=1).T
        np.testing.assert_array_equal(array, concat_array)
        # this may fail if the seek behavior is wrong
        array = handler.get(0, handler.tot_length/2)
        np.testing.assert_array_equal(array, concat_array[:concat_array.shape[0]//2])
