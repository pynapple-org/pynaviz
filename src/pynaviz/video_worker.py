from multiprocessing import Event, Queue, shared_memory

import numpy as np

from .video_handling import VideoHandler  # Replace with actual path


class RenderTriggerSource:
    """Enumeration of the renderer draw triggering source."""
    UNKNOWN = 0
    INITIALIZATION = 1
    SYNC_EVENT = 2
    LOCAL_KEY = 3


def video_worker_process(
    video_path: str,
    shape: tuple,
    shm_frame_name: str,
    shm_index_name: str,
    request_queue: Queue,
    done_event: Event,
    response_queue: Queue,
):
    handler = VideoHandler(video_path)
    shm_frame = shared_memory.SharedMemory(name=shm_frame_name)
    shm_index = shared_memory.SharedMemory(name=shm_index_name)
    frame_buffer = np.ndarray(shape, dtype=np.float32, buffer=shm_frame.buf)
    index_buffer = np.ndarray((1,), dtype=np.float32, buffer=shm_index.buf)

    while True:
        idx, move_key_frame, request_type = request_queue.get()
        print("worker", idx, move_key_frame, request_type)
        if idx is None:
            break  # Graceful shutdown        if move_key_frame is not None:

        elif request_type == RenderTriggerSource.LOCAL_KEY:
            frame, idx = handler.get_key_frame(move_key_frame)
            np.copyto(index_buffer, idx)
        else:
            frame = handler[idx]  # shape: (H, W, 3) in RGB, float32
        np.copyto(frame_buffer, frame)  # write into shared buffer
        done_event.set()  # notify main process
        response_queue.put((int(idx), request_type))  # send back both
