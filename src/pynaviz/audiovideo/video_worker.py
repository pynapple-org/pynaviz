import queue
from multiprocessing import Event, Lock, Queue, shared_memory

import numpy as np

from ..utils import RenderTriggerSource
from .video_handling import VideoHandler


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

    try:
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
                warning_msg = "[video_worker_process] Received None index, skipping frame retrieval."
                print(warning_msg)
                continue

            if request_type == RenderTriggerSource.LOCAL_KEY:
                frame, idx = handler._get_key_frame(move_key_frame)
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
    finally:
        try:
            handler.close()
        except Exception as e:
            print(f"[video_worker_process] Failed to close handler: {e}")
        try:
            shm_frame.close()
        except Exception:
            print(f"[video_worker_process] Failed to close shm: {shm_frame_name}")
        try:
            shm_index.close()
        except Exception:
            print(f"[video_worker_process] Failed to close shm: {shm_frame_name}")
