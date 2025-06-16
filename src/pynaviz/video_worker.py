from multiprocessing import shared_memory, Event, Queue
import numpy as np
import time
from .video_handling import VideoHandler  # Replace with actual path

def video_worker_main(
    video_path: str,
    shape: tuple,
    shm_name: str,
    request_queue: Queue,
    done_event: Event,
):
    handler = VideoHandler(video_path)
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

    while True:
        idx = request_queue.get()
        if idx is None:
            break  # Graceful shutdown

        frame = handler[idx]  # shape: (H, W, 3) in RGB, float32
        np.copyto(frame_buffer, frame)  # write into shared buffer
        done_event.set()  # notify main process
