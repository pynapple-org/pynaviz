import multiprocessing as mp
import queue
from multiprocessing import Event, Lock, Queue, shared_memory

import numpy as np
import pytest

from pynaviz.audiovideo.video_worker import video_worker_process
from pynaviz.utils import RenderTriggerSource


@pytest.fixture
def test_video_path():
    """
    Path to existing test video file.
    Adjust the extension if needed (.mp4, .avi, etc.)
    """
    import pathlib
    video_path = pathlib.Path(__file__).parent / "test_video/numbered_video.mp4"

    if not video_path.exists():
        pytest.skip(f"Test video not found at {video_path}")

    return str(video_path)


@pytest.fixture
def video_config(test_video_path):
    """
    Basic configuration for video testing.
    Gets the actual shape from the test video.
    """
    import av

    with av.open(test_video_path) as container:
        stream = container.streams.video[0]
        # Get first frame to determine shape
        for frame in container.decode(stream):
            height = frame.height
            width = frame.width
            break

    return {
        'video_path': test_video_path,
        'shape': (height, width, 3),  # H, W, C
    }


@pytest.fixture
def shared_memory_buffers(video_config):
    """
    Creates shared memory buffers and cleans them up after the test.

    Uses yield for automatic cleanup.
    """
    shape = video_config['shape']

    # Calculate memory sizes
    frame_size = np.prod(shape) * 4  # float32 = 4 bytes
    index_size = 1 * 4

    # Create shared memory
    shm_frame = shared_memory.SharedMemory(create=True, size=frame_size)
    shm_index = shared_memory.SharedMemory(create=True, size=index_size)

    # Provide to test
    yield {
        'shm_frame': shm_frame,
        'shm_index': shm_index,
        'frame_size': frame_size,
        'index_size': index_size,
    }

    # Cleanup (runs after test completes)
    try:
        shm_frame.close()
        shm_frame.unlink()
    except Exception as e:
        print(f"Warning: Failed to cleanup frame memory: {e}")

    try:
        shm_index.close()
        shm_index.unlink()
    except Exception as e:
        print(f"Warning: Failed to cleanup index memory: {e}")


@pytest.fixture
def mp_primitives():
    """
    Creates multiprocessing communication primitives.
    """
    return {
        'request_queue': Queue(),
        'response_queue': Queue(),
        'frame_ready': Event(),
        'stop_event': Event(),
        'buffer_lock': Lock(),
    }


@pytest.mark.parametrize("stop_event_type", ["queue", "event"])
def test_worker_starts_and_stops_cleanly(
        video_config,
        shared_memory_buffers,
        mp_primitives,
        stop_event_type,
):
    """
    Start and stop cleanly.

    Uses REAL test video file from test_video/ directory.

    This verifies:
    1. The worker process starts successfully
    2. VideoHandler can open the test video
    3. The worker responds to shutdown signal
    4. The worker exits cleanly (exit code 0)
    """
    print("\n=== TEST 1: Start and Stop ===")
    print(f"Using video: {video_config['video_path']}")

    # Start the worker process
    process = mp.Process(
        target=video_worker_process,
        args=(
            video_config['video_path'],
            video_config['shape'],
            shared_memory_buffers['shm_frame'].name,
            shared_memory_buffers['shm_index'].name,
            mp_primitives['request_queue'],
            mp_primitives['frame_ready'],
            mp_primitives['response_queue'],
            mp_primitives['stop_event'],
            mp_primitives['buffer_lock'],
        )
    )

    process.start()
    print(f"✓ Worker started (PID: {process.pid})")

    try:
        # Verify process is running
        assert process.is_alive(), "Worker should be alive after start"
        print("✓ Worker is running")

        if stop_event_type == "queue":
            # Send shutdown signal: (None, None, None)
            mp_primitives['request_queue'].put((None, None, None))
            print("✓ Shutdown signal sent")
        elif stop_event_type == "event":
            mp_primitives['stop_event'].set()
            print("✓ Shutdown signal sent")

        # Wait for process to finish (max 2 seconds)
        process.join(timeout=2.0)

        # Verify it stopped
        assert not process.is_alive(), "Worker should stop after shutdown signal"
        print("✓ Worker stopped gracefully")

        # Verify clean exit
        assert process.exitcode == 0, f"Worker should exit with 0, got {process.exitcode}"
        print("✓ Worker exited cleanly")

    finally:
        # Safety: ensure process is terminated even if test fails
        if process.is_alive():
            print("! Forcefully terminating worker")
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()


@pytest.mark.parametrize(
    "trigger_source",
    RenderTriggerSource
)
def test_worker_processes_frame_request(
        video_config,
        shared_memory_buffers,
        mp_primitives,
        trigger_source,
):
    """
    Test that the worker correctly process a frame request.

    Parametrized to test all RenderTriggerSource types.

    When trigger_source == LOCAL_KEY:
        - Worker calls handler._get_key_frame(move_key_frame)
        - This jumps to the next keyframe (index might change)
        - We verify the index was updated but don't check exact value

    For other trigger sources:
        - Worker calls handler[idx] directly
        - Index should remain the same

    This verifies:
    1. Worker receives frame request from queue
    2. Worker fetches the frame using VideoHandler
    3. Worker writes frame to shared memory
    4. Worker writes index to shared memory
    5. Worker sets the frame_ready Event
    6. Worker puts trigger in response_queue
    """
    print(f"\n=== TEST 2: Process Frame Request (trigger={trigger_source}) ===")

    # Start worker
    process = mp.Process(
        target=video_worker_process,
        args=(
            video_config['video_path'],
            video_config['shape'],
            shared_memory_buffers['shm_frame'].name,
            shared_memory_buffers['shm_index'].name,
            mp_primitives['request_queue'],
            mp_primitives['frame_ready'],
            mp_primitives['response_queue'],
            mp_primitives['stop_event'],
            mp_primitives['buffer_lock'],
        )
    )

    process.start()
    print("✓ Worker started")

    try:
        # Request frame at index 5
        requested_idx = 5
        move_key_frame = None  # Could be 1 or -1 for LOCAL_KEY to test forward/back

        print(f"Requesting frame at index {requested_idx}")
        mp_primitives['request_queue'].put((requested_idx, move_key_frame, trigger_source))

        # Wait for frame_ready event (max 2 seconds)
        is_ready = mp_primitives['frame_ready'].wait(timeout=2.0)
        assert is_ready, "frame_ready Event should be set within timeout"
        print("✓ frame_ready Event was set")

        # Check response queue for trigger
        try:
            response = mp_primitives['response_queue'].get(timeout=1.0)
            assert response == trigger_source, f"Expected {trigger_source}, got {response}"
            print(f"✓ Response queue contains correct trigger: {response}")
        except queue.Empty:
            pytest.fail("response_queue should contain trigger source")

        # Read from shared memory and verify we got a frame
        with mp_primitives['buffer_lock']:
            frame_buffer = np.ndarray(
                video_config['shape'],
                dtype=np.float32,
                buffer=shared_memory_buffers['shm_frame'].buf
            )
            index_buffer = np.ndarray(
                (1,),
                dtype=np.float32,
                buffer=shared_memory_buffers['shm_index'].buf
            )

            # Verify frame is not all zeros (actual video data)
            assert not np.all(frame_buffer == 0), "Frame should contain video data"
            print("✓ Frame contains data (not all zeros)")

            # Verify frame values are in valid range [0, 1] for float32
            assert np.all(frame_buffer >= 0) and np.all(frame_buffer <= 1), \
                "Frame values should be in range [0, 1]"
            print("✓ Frame values in valid range [0, 1]")

            # Verify index was written
            returned_idx = index_buffer[0]
            if trigger_source == RenderTriggerSource.LOCAL_KEY:
                # For LOCAL_KEY, index might jump to keyframe
                # Just verify it's a valid index (>= 0)
                assert returned_idx >= 0, f"Index should be >= 0, got {returned_idx}"
                print(f"✓ Index updated to keyframe: {returned_idx}")
            else:
                # For other triggers, index should match request
                assert returned_idx == requested_idx, \
                    f"Index should be {requested_idx}, got {returned_idx}"
                print(f"✓ Index correctly set to {requested_idx}")

    finally:
        # Shutdown
        mp_primitives['request_queue'].put((None, None, None))
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join()


@pytest.mark.parametrize(
    "trigger_source",
    RenderTriggerSource
)
def test_worker_processes_frame_last_request(
        video_config,
        shared_memory_buffers,
        mp_primitives,
        trigger_source,
):
    """
    Worker return last requested frame.

    Parametrized to test all RenderTriggerSource types.

    When trigger_source == LOCAL_KEY:
        - Worker calls handler._get_key_frame(move_key_frame)
        - This jumps to the next keyframe (index might change)
        - We verify the index was updated but don't check exact value

    For other trigger sources:
        - Worker calls handler[idx] directly
        - Index should remain the same

    This verifies:
    1. Worker receives frame request from queue
    2. Worker fetches the frame using VideoHandler
    3. Worker writes frame to shared memory
    4. Worker writes index to shared memory
    5. Worker sets the frame_ready Event
    6. Worker puts trigger in response_queue
    """
    print(f"\n=== TEST 2: Process Frame Request (trigger={trigger_source}) ===")

    # Start worker
    process = mp.Process(
        target=video_worker_process,
        args=(
            video_config['video_path'],
            video_config['shape'],
            shared_memory_buffers['shm_frame'].name,
            shared_memory_buffers['shm_index'].name,
            mp_primitives['request_queue'],
            mp_primitives['frame_ready'],
            mp_primitives['response_queue'],
            mp_primitives['stop_event'],
            mp_primitives['buffer_lock'],
        )
    )

    process.start()
    print("✓ Worker started")

    try:
        # Request frame at index 5
        requested_idx = 5
        move_key_frame = None  # Could be 1 or -1 for LOCAL_KEY to test forward/back

        print(f"Send multiple requests very quickly, last request being frame {requested_idx}.")
        for i in [3, 2, 1, 0]:
            mp_primitives['request_queue'].put((requested_idx - i, move_key_frame, trigger_source))

        # Wait for frame_ready event (max 2 seconds)
        is_ready = mp_primitives['frame_ready'].wait(timeout=2.0)
        assert is_ready, "frame_ready Event should be set within timeout"
        print("✓ frame_ready Event was set")

        # Read from shared memory and verify we got a frame
        with mp_primitives['buffer_lock']:
            frame_buffer = np.ndarray(
                video_config['shape'],
                dtype=np.float32,
                buffer=shared_memory_buffers['shm_frame'].buf
            )
            index_buffer = np.ndarray(
                (1,),
                dtype=np.float32,
                buffer=shared_memory_buffers['shm_index'].buf
            )

            # Verify frame is not all zeros (actual video data)
            assert not np.all(frame_buffer == 0), "Frame should contain video data"
            print("✓ Frame contains data (not all zeros)")

            # Verify frame values are in valid range [0, 1] for float32
            assert np.all(frame_buffer >= 0) and np.all(frame_buffer <= 1), \
                "Frame values should be in range [0, 1]"
            print("✓ Frame values in valid range [0, 1]")

            # Verify index was written
            returned_idx = index_buffer[0]
            if trigger_source == RenderTriggerSource.LOCAL_KEY:
                # For LOCAL_KEY, index might jump to keyframe
                # Just verify it's a valid index (>= 0)
                assert returned_idx >= 0, f"Index should be >= 0, got {returned_idx}"
                print(f"✓ Index updated to keyframe: {returned_idx}")
            else:
                # For other triggers, index should match request
                assert returned_idx == requested_idx, \
                    f"Index should be {requested_idx}, got {returned_idx}"
                print(f"✓ Index correctly set to the last request: {requested_idx}")

    finally:
        # Shutdown
        mp_primitives['request_queue'].put((None, None, None))
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join()
