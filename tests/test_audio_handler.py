import pathlib

import av
import numpy as np
import pytest

from pynaviz.audiovideo import audio_handling

@pytest.fixture(scope="module")
def audio_info(request):
    extension = request.param
    audio = pathlib.Path(__file__).parent / f"test_audio/sine_audio.{extension}"