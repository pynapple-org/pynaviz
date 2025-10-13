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
    audio = pathlib.Path(__file__).parent / f"test_audio/noise_audio.{extension}"
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
        assert handler.shape == concat_array.shape, ("the audio shape doesn't match expectation.\n"
                                                     f"Actual shape {handler.shape},  expected shape "
                                                     f"{concat_array.shape}")
        assert handler.tot_length == 2, "The audio duration in sec doesn't match expectation."
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
        array = handler.get(0, handler.tot_length / 2)
        np.testing.assert_array_equal(array, concat_array[:concat_array.shape[0]//2])
        array = handler.get(handler.tot_length / 2, handler.tot_length)
        np.testing.assert_array_equal(array, concat_array[concat_array.shape[0] // 2:])
        array = handler.get(handler.tot_length / 4., handler.tot_length - handler.tot_length / 4.)
        np.testing.assert_array_equal(array, concat_array[concat_array.shape[0] // 4: - concat_array.shape[0] // 4])


@pytest.mark.parametrize("fully_decoded_audio", ["wav", "mp3", "flac"], indirect=True)
def test_decode_first_boundaries(fully_decoded_audio):
    audio_path, frame_arrays, frame_pts, frame_av, frame_size = fully_decoded_audio

    with audio_handling.AudioHandler(audio_path) as handler:
        # Preload so current_frame is set
        _ = handler.get(0, 0.1)  # decode first chunk
        cf = handler.current_frame
        assert cf is not None

        # Case 1: start == cf.pts
        frames, _ = handler._decode_first(cf.pts)
        # Should return exactly from that PTS forward
        expected = cf.to_ndarray()
        np.testing.assert_array_equal(frames[0], expected)

        # Case 2: start == cf.pts + cf.duration
        start_at_next = cf.pts + cf.duration
        # Make sure that we are still at cf frame
        assert handler.current_frame == cf
        frames, _ = handler._decode_first(start_at_next)
        # This should start at the *next* frame
        end_bound = cf.pts + cf.duration
        frames, cur = handler._decode_first(end_bound)
        assert frames[0].shape[1] == 0  # empty first piece
        assert cur == end_bound


@pytest.mark.parametrize("fully_decoded_audio", ["wav", "mp3", "flac"], indirect=True)
def test_start_end_order(fully_decoded_audio):
    audio_path = fully_decoded_audio[0]
    with audio_handling.AudioHandler(audio_path) as handler:
        with pytest.raises(ValueError, match="`end` time must be greater"):
            handler.get(1.8, 1)


@pytest.mark.parametrize("fully_decoded_audio", ["wav", "mp3", "flac"], indirect=True)
def test_start_greater_than_file_len(fully_decoded_audio):
    audio_path = fully_decoded_audio[0]
    with audio_handling.AudioHandler(audio_path) as handler:
        data = handler.get(10, 11)
        assert data.shape == (0, 1)
