"""

"""
import pathlib
import sys

import pytest

import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


@pytest.fixture
def stream(dummy_tsdframe):
    func = lambda x: x
    return viz.threads.data_streaming.TsdFrameStreaming(dummy_tsdframe, func, 1.0)


def test_tsdframe_init(dummy_tsdframe, stream):
    assert stream.data is dummy_tsdframe
    assert stream.window_size == 1.0
    assert stream._max_n > 0
    assert not stream._flushed
    assert stream._max_n == 101
    assert stream._slice_ == slice(0, 100)

@pytest.mark.parametrize(
    "start, end",
    [
        (0.0, 1.0),
        (0.5, 1.5),
        (5.0, 6.0),
        (99.0, 100.0),
        (-10.0, -9.0),
        (200.0, 201.0),
    ]
)
def test_tsdframe_get_slice(dummy_tsdframe, stream, start, end):
    slice_ = stream.get_slice(start, end)
    width = end - start
    nap_slice = dummy_tsdframe._get_slice(start-width, end+width, n_points=int(stream._max_n))
    assert slice_ == nap_slice

def test_tsdframe_get_slice_errors(stream):
    with pytest.raises(ValueError):
        stream.get_slice(1.0, 0.0)

def test_tsdframe_callback(stream):
    calls = []
    stream._callback = lambda s: calls.append(s)
    stream._callback(slice(0, 10))
    assert calls == [slice(0, 10)]

@pytest.mark.parametrize(
    "width",
    [0.5, 1.0, 2.0]
)
def test_tsdframe_stream(dummy_tsdframe, stream, width):
    calls = []
    slices = []
    stream._callback = lambda s: calls.append(s)
    # Simulate streaming over the entire range of the data
    position = 0
    step = 1
    while position < dummy_tsdframe.t[-1]:
        stream.stream((position, 0), width)
        start, end = position - width / 2, position + width / 2
        slices.append(dummy_tsdframe._get_slice(start-width, end+width, n_points=int(stream._max_n)))
        position += step

    assert len(calls) > 0
    for call, expected in zip(calls, slices):
        assert call == expected

def test_tsdframe_stream_twice(dummy_tsdframe, stream):
    calls = []
    stream._callback = lambda s: calls.append(s)
    assert not stream._flushed
    stream.stream((0, 0), 1.0)
    assert stream._flushed
    first_call = calls[-1]
    stream.stream((0, 0), 1.0)
    second_call = calls[-1]
    assert first_call == second_call
    stream._flushed = False
    stream.stream((0, 0), 1.0)
    assert calls[-1] == first_call
    assert stream._flushed

def test_tsdframe_stream_len(dummy_tsdframe, stream):
    assert hasattr(stream, '__len__')
    assert len(stream.data) == len(dummy_tsdframe)
    assert len(stream) == 101
