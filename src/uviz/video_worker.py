import queue
from enum import Enum
from multiprocessing import Event, Lock, Queue, shared_memory

import numpy as np

from .video_handling import VideoHandler


class RenderTriggerSource(Enum):
    """Enumeration of the renderer draw triggering source."""

    UNKNOWN = 0
    INITIALIZATION = 1
    ZOOM_TO_POINT = 2
    SYNC_EVENT_RECEIVED = 3
    LOCAL_KEY = 4
    SET_FRAME = 5

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


def video_worker_process(
    video_path: str,
    shape: tuple,
    shm_frame_name: str,
    shm_index_name: str,
    request_queue: Queue,
    frame_ready: Event,
    response_queue: Queue,
    stop_event: Event,
    buffer_lock: Lock
):
    handler = VideoHandler(video_path)
    shm_frame = shared_memory.SharedMemory(name=shm_frame_name)
    shm_index = shared_memory.SharedMemory(name=shm_index_name)
    frame_buffer = np.ndarray(shape, dtype=np.float32, buffer=shm_frame.buf)
    index_buffer = np.ndarray((1,), dtype=np.float32, buffer=shm_index.buf)

    while not stop_event.is_set():
        try:
            # wait for a new request
            item = request_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # if we received a shutdown signal terminate
        if item[0] is None:
            break

        # empty the queue keeping the most recent item
        while True:
            try:
                latest = request_queue.get_nowait()
                if latest[0] is None:
                    # shutdown signal received, break immediately
                    item = latest
                    break
                item = latest
            except queue.Empty:
                break

        # unpack latest request
        idx, move_key_frame, request_type = item

        # TODO: unsure if this can happen now that i have the event
        if idx is None:
            break

        if request_type == RenderTriggerSource.LOCAL_KEY:
            frame, idx = handler.get_key_frame(move_key_frame)
        else:
            frame = handler[idx]  # shape: (H, W, 3) in RGB, float32

        with buffer_lock:
            np.copyto(frame_buffer, frame)
            np.copyto(index_buffer, idx)

            # drain response_queue to remove stale triggers
            while True:
                try:
                    _ = response_queue.get_nowait()
                except queue.Empty:
                    break

            # only now enqueue the trigger
            response_queue.put(request_type)
        frame_ready.set()
    try:
        handler.close()
    except Exception as e:
        print(f"[video_worker_process] Failed to close handler: {e}")
