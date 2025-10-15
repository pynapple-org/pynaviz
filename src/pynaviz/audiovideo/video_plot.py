"""
Module for time-synchronized video plotting using pygfx and multiprocessing.

"""

import abc
import atexit
import pathlib
import queue
import signal
import sys
import threading
import weakref
from abc import ABC, abstractmethod
from multiprocessing import Event, Process, Queue, current_process, set_start_method, shared_memory
from multiprocessing import Lock as MultiProcessLock
from typing import Any, Optional

import av
import numpy as np
import pygfx as gfx
import pynapple as nap
from numpy.typing import NDArray

from ..base_plot import _BasePlot
from ..controller import GetController
from ..utils import GRADED_COLOR_LIST
from .skeleton_plot import PlotPoints
from .video_handling import VideoHandler
from .video_worker import RenderTriggerSource, video_worker_process

# WeakSet to avoid keeping dead references
_active_plot_videos = weakref.WeakSet()


def _cleanup_all_plot_videos():
    """Cleans up all active video plot instances on exit."""
    for video in list(_active_plot_videos):
        try:
            video.close()
        except Exception as e:
            print(f"[WARN] Error during close: {e}")
    _active_plot_videos.clear()


# Register cleanup at process exit
def _register_cleanup_hooks():
    """Register cleanup hooks only in the main process."""
    if current_process().name != "MainProcess":
        return
    if getattr(_register_cleanup_hooks, "_registered", False):
        return
    atexit.register(_cleanup_all_plot_videos)
    signal.signal(signal.SIGINT, lambda *_: (_cleanup_all_plot_videos(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup_all_plot_videos(), sys.exit(0)))
    _register_cleanup_hooks._registered = True

if sys.platform != "win32":
    try:
        set_start_method("fork", force=True)
    except RuntimeError:
        pass


def _update_buffer(plot_object: Any, frame_index: int):
    """Update the texture buffer of a plot object for a specific frame."""
    if (
            plot_object.texture.data.shape[0] == 1 and plot_object.texture.data.shape[1] == 3
    ):  # assume single point
        plot_object.texture.data[0, 0:2] = plot_object.data.values[frame_index].astype("float32")
    else:
        img_array = plot_object.data.values[frame_index]
        plot_object.texture.data[:] = img_array.astype("float32")
    plot_object.texture.update_full()
    plot_object._set_time_text(frame_index)
    return


class PlotBaseVideoTensor(_BasePlot, ABC):
    """
    Abstract base class for time-synchronized video plots using pygfx.

    Subclasses must implement the methods to provide video frames as 2D tensors.
    """

    def __init__(self, data: Any, index: Optional[int] = None, parent: Optional[Any] = None) -> None:
        """
        Initialize the base video tensor plot.

        Parameters
        ----------
        data : Any
            The data source, such as a TsdTensor or a video handler.
        index : int, optional
            Identifier for the controller.
        parent : Any, optional
            Parent widget or container.
        """
        super().__init__(data, parent=parent, maintain_aspect=True)

        texture_data = self._get_initial_texture_data()
        self.texture = gfx.Texture(texture_data.astype("float32"), dim=2)
        self.image = gfx.Image(
            gfx.Geometry(grid=self.texture),
            gfx.ImageBasicMaterial(clim=(0, 1)),
        )

        img_height = self.texture.size[1]
        self.time_text = gfx.Text(
            text="0.0",
            font_size=img_height * 0.05,
            anchor="bottom-left",
            material=gfx.TextMaterial(
                color="#B4F8C8", outline_color="#000", outline_thickness=0.15
            ),
        )
        self.time_text.geometry.anchor = "bottom-left"

        self.scene.add(self.image, self.time_text)
        self.camera.show_object(self.scene)

        self.controller = GetController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            data=self._data,
            buffer=self.texture,
            callback=self._update_buffer,
        )

        # List to hold time series points that can be superposed
        self.points = {}

        self.canvas.request_draw(
            draw_function=lambda: self.renderer.render(self.scene, self.camera)
        )

    @abstractmethod
    def _get_initial_texture_data(self) -> np.ndarray:
        """Return the initial 2D image tensor for the texture."""
        pass

    def _set_time_text(self, frame_index: int):
        """Update the on-screen time text based on the current frame index."""
        if self.time_text:
            self.time_text.set_text(str(np.round(self.data.t[frame_index], 4)))

    def set_frame(self, target_time: float):
        """
        Set the video display to the frame closest to the target time.

        Parameters
        ----------
        target_time : float
            Time in seconds to seek to.
        """
        self.controller.set_frame(target_time)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
        """Placeholder for future metadata sorting method."""
        pass

    def group_by(self, metadata_name: str, spacing: Optional = None):
        """Placeholder for future metadata grouping method."""
        pass

    @abc.abstractmethod
    def _update_buffer(self, frame_index: int, event_type: Optional[RenderTriggerSource] = None):
        """
        Abstract method to update the buffer based on the frame index and event.

        Parameters
        ----------
        frame_index : int
            Index of the frame to load.
        event_type : RenderTriggerSource, optional
            Source of the event triggering the update.
        """
        pass

    def _update_extra_objects(self, frame_index: int, event_type: Optional[RenderTriggerSource] = None):
        """
        Update the positions of any superposed time series points based on the current frame.

        Parameters
        ----------
        frame_index : int
            Current frame index.
        event_type : RenderTriggerSource, optional
            Source of the event triggering the update.
        """
        if len(self.points):
            time_array = getattr(self.data.index, "values", self.data.index)
            current_time = time_array[frame_index]
            for points in self.points.values():
                points.update(current_time)

    def superpose_points(self, tsdframe: nap.TsdFrame,
                         color = None,
                         markersize: float = 10,
                         thickness: float = 2,
                         label: Optional[str] = None
                         ):
        """
        Superpose a set of points on top of the video plot. Argument tsdframe should
        be a nap.Tsdframe with multiple of 2 columns (x1, y1, x2, y2, ...).
        The label of the columns doesn't matter.

        Parameters
        ----------
        tsdframe : nap.TsdFrame
            Time series data to overlay.
        color : str or rgba tuple, optional
            Color of the set of points.
        markersize : float, default=10
            Size of the set of points.
        thickness : float, default=2
            Thickness of the lines connecting the points.
        label : str, optional
            Label for the set of points. If None, a default label is assigned.
        """
        if not isinstance(tsdframe, nap.TsdFrame):
            raise ValueError("tsdframe must be a nap.TsdFrame instance.")

        if tsdframe.shape[1] % 2 != 0:
            raise ValueError("tsdframe must have multiple of 2 columns for x and y coordinates.")

        if label is None:
            label = f"points_{len(self.points) + 1}"

        if color is None:
            color = GRADED_COLOR_LIST[len(self.points) % len(GRADED_COLOR_LIST)]

        self.points[label] = PlotPoints(
                tsdframe = tsdframe,
                initial_time = self.controller._get_frame_time(),
                scene=self.scene,
                color=color,
                markersize = markersize,
                thickness = thickness
            )


        self.controller.renderer_request_draw()


class PlotTsdTensor(PlotBaseVideoTensor):
    """
    Visualization for 3-dimensional time series (``nap.TsdTensor``) displayed as movies.
    """

    def __init__(self, data: nap.TsdTensor, index=None, parent=None):
        self._data = data
        super().__init__(data, index=index, parent=parent)
        self.controller._add_callback(self._update_extra_objects)

    def _get_initial_texture_data(self):
        """Return the first frame as the initial texture."""
        return self._data.values[0]

    def _update_buffer(self, frame_index, event_type: Optional[RenderTriggerSource] = None):
        """Synchronously update buffer for the given frame index."""
        _update_buffer(self, frame_index)
        self.controller.renderer_request_draw()


class PlotVideo(PlotBaseVideoTensor):
    """
    Visualization for rendering video files synchronized with a time series.

    This class uses shared memory and multiprocessing to efficiently stream frames
    from disk to GPU via pygfx. It also supports real-time interaction and frame-based control.
    """

    _debug = False

    def __init__(
        self,
        video: str | pathlib.Path | VideoHandler,
        t: Optional[NDArray] = None,
        stream_index: int = 0,
        index=None,
        start_worker: bool = True,
        parent=None,
    ):
        """
        Initialize the PlotVideo instance with a given video source.

        Parameters
        ----------
        video : str or pathlib.Path or VideoHandler
            Path to the video file to be visualized or a VideoHandler object.
        t : NDArray, optional
            Time vector to use for syncing frames.
        stream_index : int, default=0
            Index of the stream to read in the video file.
        index : int, optional
            Controller ID index.
        start_worker : bool, optional
            Start the worker thread. This should be set to true when used in GUI but for
            speeding up tests, should be set to False.
        parent : object, optional
            Parent GUI container or widget.
        """
        self._closed = False
        if isinstance(video, (str, pathlib.Path)):
            data = VideoHandler(video, time=t, stream_index=stream_index)
        else:
            if not isinstance(video, VideoHandler):
                raise ValueError("video must be a file path or a VideoHandler instance.")
            data = video
        self._data = data
        video_path = data.file_path
        super().__init__(data, index=index, parent=parent)

        # Shared memory setup for multiprocessing frame exchange
        self.shape = self.texture.data.shape
        self._pending_ui_update_queue = queue.Queue()
        self._stop_threads = threading.Event()

        self.buffer_lock = threading.Lock()
        self._needs_redraw = threading.Event()

        if start_worker:
            # Queues and events for IPC
            self._buffer_thread = threading.Thread(target=self._update_buffer_thread, daemon=True)
            self.shm_frame = shared_memory.SharedMemory(
                create=True, size=np.prod(self.shape) * np.float32().nbytes
            )
            self.shm_index = shared_memory.SharedMemory(create=True, size=np.float32().nbytes)
            self.shared_frame = np.ndarray(self.shape, dtype=np.float32, buffer=self.shm_frame.buf)
            self.shared_index = np.ndarray(shape=(1,), dtype=np.float32, buffer=self.shm_index.buf)

            self.request_queue = Queue()
            self.response_queue = Queue()
            self.frame_ready = Event()
            self.worker_stop_event = Event()

            self.renderer.add_event_handler(self._move_fast, "key_down")

            # Worker process to read video frames asynchronously
            self.worker_lock = MultiProcessLock()
            self._worker = Process(
                target=video_worker_process,
                args=(
                    video_path,
                    self.shape,
                    self.shm_frame.name,
                    self.shm_index.name,
                    self.request_queue,
                    self.frame_ready,
                    self.response_queue,
                    self.worker_stop_event,
                    self.worker_lock,
                ),
                daemon=False,
            )

            self._worker.start()
            self._buffer_thread.start()

        # Registry and buffer setup
        _register_cleanup_hooks()
        _active_plot_videos.add(self)

        self.canvas.request_draw(self.animate)
        self._last_jump_index = 0

    def _get_initial_texture_data(self):
        """Return the first video frame as the initial texture."""
        return self._data.get(self._data.time[0])

    @property
    def data(self):
        """Read-only access to the video data handler."""
        return self._data

    @data.setter
    def data(self, value):
        raise ValueError("Cannot set data for ``PlotVideo``. Data must be a fixed video stream.")

    def close(self):
        """Cleanly close shared memory, worker, and background thread."""
        if not self._closed:
            try:
                try:
                    self._data.close()
                except Exception as e:
                    print(f"Unable to close VideoHandler with exception: {e}")

                try:
                    self._stop_threads.set()
                    if hasattr(self, "_buffer_thread") and self._buffer_thread.is_alive():
                        self._buffer_thread.join(timeout=1)
                except Exception as e:
                    print(f"Unable to stop background thread with exception: {e}")

                if hasattr(self, "worker_stop_event") and self._worker.is_alive():
                    try:
                        self.worker_stop_event.set()
                        self._worker.join(timeout=2)
                    except Exception as e:
                        print(f"Unable to stop worker process with exception: {e}")

                    # After closing and unlinking shared memory
                    if hasattr(self, "shm_frame") and self.shm_frame is not None:
                        try:
                            self.shm_frame.close()
                            self.shm_frame.unlink()
                            # drop all references to shm
                            # otherwise a resource manager process will hang
                            self.shared_frame = None
                            self.shm_frame = None
                        except Exception as e:
                            print(f"[ERROR] Unable to close shm_frame: {e}")

                    if hasattr(self, "shm_index") and self.shm_index is not None:
                        try:
                            self.shm_index.close()
                            self.shm_index.unlink()
                            # drop all references to shm
                            # otherwise a resource manager process will hang
                            self.shm_index = None
                            self.shared_index = None
                        except Exception as e:
                            print(f"[ERROR] Unable to close shm_index: {e}")
            finally:
                _active_plot_videos.discard(self)
                self._closed = True
                super().close()

    def _move_fast(self, event, delta=1):
        """
        Jump between frames using arrow keys.

        Parameters
        ----------
        event : pygfx.KeyEvent
            The key event that was triggered.
        delta : int, default=1
            Step to move forward or backward.
        """
        if event.type == "key_down" and event.key in ("ArrowRight", "ArrowLeft"):
            self.frame_ready.clear()
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except queue.Empty:
                    break

            self.request_queue.put((False, event.key == "ArrowLeft", RenderTriggerSource.LOCAL_KEY))
            self._last_jump_index = self.controller.frame_index

    def _update_buffer(self, frame_index, event_type: Optional[RenderTriggerSource] = None):
        """
        Update the video buffer based on the event type.

        Parameters
        ----------
        frame_index : int
            Index of the frame to display.
        event_type : RenderTriggerSource, optional
            Source of the triggering event.
        """
        if self._debug:
            print(f"update buffer: {id(self)}({id(self.controller)}) - {event_type} - {frame_index}")
        # Async playing
        if event_type != RenderTriggerSource.SET_FRAME:
            self.frame_ready.clear()
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except queue.Empty:
                    break
            event_type = event_type or RenderTriggerSource.UNKNOWN
            self.request_queue.put((frame_index, None, event_type))
            self._last_requested_frame_index = frame_index
        # Sync jump to frame
        else:
            frame = self.data[frame_index]
            if isinstance(frame, av.VideoFrame):
                frame = frame.to_ndarray(format="rgb24")[::-1] / 255.0
            with self.buffer_lock:
                self.texture.data[:] = frame
                self._set_time_text(frame_index)
                self.texture.update_full()
                self._update_extra_objects(frame_index)

    def _update_buffer_thread(self):
        """Background thread that listens for ready frames and updates the buffer."""
        while not self._stop_threads.is_set():
            if not self.frame_ready.wait(timeout=0.1):
                continue
            with self.buffer_lock, self.worker_lock:
                self.texture.data[:] = self.shared_frame
                frame_index = int(self.shared_index[0])
                self._update_extra_objects(frame_index)
                self.frame_ready.clear()
                try:
                    trigger_source = self.response_queue.get(timeout=0.05)
                    while not self._pending_ui_update_queue.empty():
                        try:
                            self._pending_ui_update_queue.get_nowait()
                        except queue.Empty:
                            break
                    self._pending_ui_update_queue.put((frame_index, trigger_source))

                except queue.Empty:
                    continue
            self._needs_redraw.set()

    def __del__(self):
        """Ensure all resources are closed when the object is destroyed."""
        self.close()

    def animate(self):
        """
        Main render loop callback for pygfx.

        Handles texture and text update, syncing, and redraw triggering.
        """
        update = False
        try:
            frame_index, trigger_source = self._pending_ui_update_queue.get_nowait()
            with self.buffer_lock:
                self._set_time_text(frame_index)
                self.texture.update_full()
            self.controller.frame_index = frame_index
            self.controller.renderer_request_draw()

            if trigger_source == RenderTriggerSource.LOCAL_KEY and hasattr(self, "_last_jump_index"):
                current_time = self.controller._get_frame_time()
                if self._debug:
                    print("keypress", current_time, frame_index)
                del self._last_jump_index
                self.controller._send_sync_event(update_type="pan", current_time=current_time)

            elif trigger_source == RenderTriggerSource.ZOOM_TO_POINT:
                current_time = self.controller._get_frame_time()
                if self._debug:
                    print("zoom", current_time, frame_index)
                self.controller._send_sync_event(update_type="pan", current_time=current_time)

        except queue.Empty:
            update = True

        if self._needs_redraw.is_set() or update:
            self.renderer.render(self.scene, self.camera)
            self._needs_redraw.clear()

        self.canvas.request_draw(self.animate)
