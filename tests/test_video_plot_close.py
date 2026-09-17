"""Tests for the non-blocking shutdown path of ``PlotVideo.close``.

``close`` used to join the worker process inline with a 2 s timeout. A worker
created moments earlier has not reached its serve loop yet -- under the "spawn"
start method it is still importing its dependencies -- so it could not observe
``worker_stop_event``, and the join burned its full timeout on the GUI thread
every time while the worker was abandoned alive anyway.

``close`` now signals the worker and hands the process, the buffer thread and
the shared memory to a background reaper thread, which joins them and releases
the memory only once the worker is really gone.
"""

import multiprocessing as mp
import pathlib
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

import pynaviz.audiovideo.video_plot as video_plot
from pynaviz.audiovideo.video_plot import PlotVideo, _submit_for_reaping, _WorkerHandles

# ``close`` still tears the canvas down synchronously (~100 ms), so allow a
# margin. The point is that it no longer waits on the worker, which cost a flat
# 2 s before. Anything at or above 2 s means the regression is back.
MAX_CLOSE_SECONDS = .5

# A stub worker lifetime deliberately longer than the old 2 s join timeout, so
# a blocking teardown cannot possibly pass the timing assertions below.
SLOW_WORKER_SECONDS = 3.0


@pytest.fixture
def test_video_path():
    video_path = pathlib.Path(__file__).parent / "test_video/numbered_video.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found at {video_path}")
    return str(video_path)


@pytest.fixture(autouse=True)
def quiet_reaper():
    """Keep the module-global reaper state from leaking between tests."""
    assert video_plot._drain_reaper(timeout=60), "reaper busy before test"
    yield
    assert video_plot._drain_reaper(timeout=60), "reaper still busy after test"


@pytest.fixture(params=["fork", "spawn"])
def start_method(request):
    """Run against every start method available on this platform.

    Windows only has "spawn", and that is where the blocking join hurt most, so
    it is covered explicitly rather than inherited from the module default.
    """
    if request.param not in mp.get_all_start_methods():
        pytest.skip(f"{request.param} start method unavailable")
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method(request.param, force=True)
    yield request.param
    # On Windows nothing has set a method at import time, so ``previous`` can
    # be None, which ``set_start_method`` rejects.
    if previous is not None:
        mp.set_start_method(previous, force=True)


def _shm_exists(name):
    """True if a shared memory block by this name can still be attached."""
    try:
        existing = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    existing.close()
    return True


# ----------------------------------------------------------------------
# The reaper mechanism, isolated from worker start-up timing
# ----------------------------------------------------------------------

def _stubborn_worker(seconds):
    """A worker that ignores the stop event, like one still importing."""
    time.sleep(seconds)


def test_handoff_does_not_wait_for_a_worker_that_ignores_the_stop_event():
    """The regression, deterministically: a worker outliving the old timeout.

    The hand-off must return immediately even though this worker stays alive
    for longer than the 2 s that ``close`` used to spend joining it.
    """
    shm = shared_memory.SharedMemory(create=True, size=64)
    name = shm.name
    view = np.ndarray((16,), dtype=np.float32, buffer=shm.buf)
    worker = mp.Process(target=_stubborn_worker, args=(SLOW_WORKER_SECONDS,))
    worker.start()

    start = time.perf_counter()
    _submit_for_reaping(_WorkerHandles(worker=worker, shm=[shm], views=[view]))
    elapsed = time.perf_counter() - start

    assert elapsed < MAX_CLOSE_SECONDS, f"hand-off blocked for {elapsed:.2f}s"
    # Returned while the worker is still running: that is the whole point.
    assert worker.is_alive()
    assert _shm_exists(name), "memory released before the worker exited"

    # The reaper, not the caller, completes the teardown.
    assert video_plot._drain_reaper(timeout=60)
    assert not worker.is_alive()
    assert worker.exitcode == 0
    assert not _shm_exists(name), "reaper did not release the shared memory"


def test_reaper_thread_is_a_daemon():
    """The reaper must never keep the interpreter alive."""
    shm = shared_memory.SharedMemory(create=True, size=64)
    _submit_for_reaping(_WorkerHandles(shm=[shm]))
    assert video_plot._drain_reaper(timeout=60)

    reaper = video_plot._reaper_thread
    assert reaper is not None and reaper.daemon
    assert reaper.name == "pynaviz-worker-reaper"
    assert reaper in threading.enumerate()


def test_reaper_releases_memory_when_there_is_no_worker():
    """A plot closed after its worker already died must not leak memory."""
    shm = shared_memory.SharedMemory(create=True, size=64)
    name = shm.name
    _submit_for_reaping(_WorkerHandles(worker=None, shm=[shm]))

    assert video_plot._drain_reaper(timeout=60)
    assert not _shm_exists(name)


def test_reaper_survives_a_failing_handoff():
    """One bad hand-off must not kill the thread and stall every later close."""
    broken = _WorkerHandles()
    broken.shm = None  # makes _release_handles raise

    _submit_for_reaping(broken)
    assert video_plot._drain_reaper(timeout=60), "reaper stalled on a failure"

    # Still functional afterwards.
    shm = shared_memory.SharedMemory(create=True, size=64)
    name = shm.name
    _submit_for_reaping(_WorkerHandles(shm=[shm]))
    assert video_plot._drain_reaper(timeout=60)
    assert not _shm_exists(name)


# ----------------------------------------------------------------------
# End to end, through a real PlotVideo
# ----------------------------------------------------------------------

def test_worker_and_shared_memory_are_created(test_video_path, start_method):
    """Guard the setup the teardown tests depend on."""
    plot = PlotVideo(video=test_video_path, t=np.arange(100))
    try:
        assert isinstance(plot._worker, mp.process.BaseProcess)
        assert plot._worker.is_alive()
        assert not plot._worker.daemon, "worker must outlive close() to detach"

        frame_bytes = int(np.prod(plot.shape)) * np.float32().nbytes
        assert plot.shm_frame.size >= frame_bytes
        assert plot.shm_index.size >= np.float32().nbytes
        # Really allocated, not just wrapped objects.
        assert _shm_exists(plot.shm_frame.name)
        assert _shm_exists(plot.shm_index.name)

        assert plot.shared_frame.shape == tuple(plot.shape)
        assert plot.shared_index.shape == (1,)
        assert plot._buffer_thread.is_alive()
    finally:
        plot.close()
        assert video_plot._drain_reaper(timeout=60)


def test_close_returns_within_threshold(test_video_path, start_method):
    """``close`` issued right after opening must not block on the worker."""
    plot = PlotVideo(video=test_video_path, t=np.arange(100))
    time.sleep(0.3)  # the issue's window: worker still coming up

    start = time.perf_counter()
    plot.close()
    elapsed = time.perf_counter() - start

    assert elapsed < MAX_CLOSE_SECONDS, f"close() blocked for {elapsed:.2f}s"
    assert plot._closed
    assert video_plot._drain_reaper(timeout=60)


def test_close_immediately_after_construction_returns_within_threshold(
    test_video_path, start_method
):
    """The worst case: no chance at all for the worker to reach its loop."""
    plot = PlotVideo(video=test_video_path, t=np.arange(100))

    start = time.perf_counter()
    plot.close()
    elapsed = time.perf_counter() - start

    assert elapsed < MAX_CLOSE_SECONDS, f"close() blocked for {elapsed:.2f}s"
    assert video_plot._drain_reaper(timeout=60)


def test_close_tears_down_worker_thread_and_memory(test_video_path, start_method):
    """After the reaper runs, nothing from the plot is left behind."""
    plot = PlotVideo(video=test_video_path, t=np.arange(100))
    worker = plot._worker
    buffer_thread = plot._buffer_thread
    names = [plot.shm_frame.name, plot.shm_index.name]

    plot.close()

    # close() drops its own handles so the reaper owns the only references.
    assert plot.shm_frame is None
    assert plot.shm_index is None
    assert plot.shared_frame is None
    assert plot.shared_index is None
    assert plot.worker_stop_event.is_set()
    assert plot._stop_threads.is_set()

    assert video_plot._drain_reaper(timeout=60)

    assert not worker.is_alive(), "worker was not joined"
    # Termination is asserted, not the exit status. Under "fork" the worker is
    # occasionally killed by SIGSEGV during start-up, because ``VideoHandler``
    # starts a libav demux thread just before the fork and the child re-enters
    # libav with that thread's mutexes copied in a locked state. That is a
    # start-up race, independent of the teardown under test here, and
    # ``test_handoff_...`` asserts a clean exit on a worker that never touches
    # libav. Reap must still be complete either way, which is what follows.
    assert worker.exitcode is not None, "worker was not reaped"
    assert not buffer_thread.is_alive(), "buffer thread was not joined"
    for name in names:
        assert not _shm_exists(name), f"shared memory {name} leaked"


def test_repeated_open_close_cycles(test_video_path, start_method):
    """Repeated cycles stay fast and leak neither processes nor memory."""
    workers, names = [], []
    for _ in range(3):
        plot = PlotVideo(video=test_video_path, t=np.arange(100))
        workers.append(plot._worker)
        names += [plot.shm_frame.name, plot.shm_index.name]
        time.sleep(0.2)  # close while the worker is still coming up

        start = time.perf_counter()
        plot.close()
        elapsed = time.perf_counter() - start
        assert elapsed < MAX_CLOSE_SECONDS, f"close() blocked for {elapsed:.2f}s"

    assert video_plot._drain_reaper(timeout=60)
    assert not any(w.is_alive() for w in workers)
    assert not any(_shm_exists(n) for n in names)


def test_close_is_idempotent(test_video_path, start_method):
    """A second close (from ``__del__`` or the atexit hook) is a no-op."""
    plot = PlotVideo(video=test_video_path, t=np.arange(100))
    worker = plot._worker
    plot.close()

    start = time.perf_counter()
    plot.close()
    assert time.perf_counter() - start < MAX_CLOSE_SECONDS

    assert video_plot._drain_reaper(timeout=60)
    assert not worker.is_alive()


def test_cleanup_hook_closes_and_drains(test_video_path, start_method):
    """The atexit hook must close live plots and wait for their workers."""
    plot = PlotVideo(video=test_video_path, t=np.arange(100))
    worker = plot._worker
    names = [plot.shm_frame.name, plot.shm_index.name]
    assert plot in video_plot._active_plot_videos

    video_plot._cleanup_all_plot_videos()

    assert plot._closed
    assert not worker.is_alive(), "hook returned with the worker still alive"
    assert not any(_shm_exists(n) for n in names)


def test_close_without_worker_does_not_submit_to_reaper(test_video_path):
    """``start_worker=False`` allocates nothing, so there is nothing to reap."""
    plot = PlotVideo(video=test_video_path, t=np.arange(100), start_worker=False)
    assert not hasattr(plot, "shm_frame")

    start = time.perf_counter()
    plot.close()
    assert time.perf_counter() - start < MAX_CLOSE_SECONDS
    assert plot._closed


# ----------------------------------------------------------------------
# Degraded paths: the teardown must stay best-effort and never raise
# ----------------------------------------------------------------------

class _StuckThread:
    """A buffer thread that refuses to stop."""

    name = "stuck-buffer-thread"

    def is_alive(self):
        return True

    def join(self, timeout=None):
        return None


class _RaisingEvent:
    """An event whose ``set`` fails, e.g. on a closed semaphore."""

    def set(self):
        raise OSError("semaphore is gone")


class _RaisingShm:
    """Shared memory that cannot be released."""

    name = "unreleasable"

    def close(self):
        raise BufferError("cannot close exported pointer")

    def unlink(self):
        raise AssertionError("must not be reached")


def test_release_leaks_memory_rather_than_freeing_it_under_a_live_thread(capsys):
    """Freeing memory a live thread still reads would crash the process."""
    shm = shared_memory.SharedMemory(create=True, size=64)
    name = shm.name
    try:
        video_plot._release_handles(
            _WorkerHandles(buffer_thread=_StuckThread(), shm=[shm])
        )
        assert "did not stop" in capsys.readouterr().out
        assert _shm_exists(name), "freed memory out from under a live thread"
    finally:
        shm.close()
        shm.unlink()


def test_release_frees_memory_when_the_worker_overruns_its_join(capsys, monkeypatch):
    """A worker that will not exit must not strand the memory forever."""
    monkeypatch.setattr(video_plot, "_WORKER_JOIN_TIMEOUT", 0.2)
    shm = shared_memory.SharedMemory(create=True, size=64)
    name = shm.name
    worker = mp.Process(target=_stubborn_worker, args=(SLOW_WORKER_SECONDS,))
    worker.start()
    try:
        video_plot._release_handles(_WorkerHandles(worker=worker, shm=[shm]))
        out = capsys.readouterr().out
        assert "did not exit within" in out
        assert not _shm_exists(name), "memory not released after giving up"
    finally:
        worker.join(timeout=60)


def test_release_reports_a_failing_block_and_still_frees_the_others(capsys):
    """One unreleasable block must not strand the rest."""
    good = shared_memory.SharedMemory(create=True, size=64)
    name = good.name

    video_plot._release_handles(_WorkerHandles(shm=[_RaisingShm(), good]))

    assert "Unable to release shared memory unreleasable" in capsys.readouterr().out
    assert not _shm_exists(name), "a failing block blocked the others"


def test_close_survives_a_handler_that_fails_to_close(test_video_path, capsys):
    """``close`` must finish even if the video handler raises."""
    plot = PlotVideo(video=test_video_path, t=np.arange(100), start_worker=False)

    def boom():
        raise RuntimeError("container already gone")

    plot._data.close = boom
    plot.close()

    assert "Unable to close VideoHandler" in capsys.readouterr().out
    assert plot._closed


def test_close_survives_a_stop_event_that_cannot_be_set(test_video_path, capsys):
    """``close`` must finish even if signalling the worker raises."""
    plot = PlotVideo(video=test_video_path, t=np.arange(100), start_worker=False)
    plot.worker_stop_event = _RaisingEvent()

    plot.close()

    assert "Unable to signal worker" in capsys.readouterr().out
    assert plot._closed
    assert plot._stop_threads.is_set(), "threads must still be told to stop"


def test_cleanup_hook_reports_a_failing_close_and_continues(test_video_path, capsys):
    """One broken plot must not stop the atexit hook closing the rest."""

    class _BrokenPlot:
        def close(self):
            raise RuntimeError("close failed")

    broken = _BrokenPlot()
    plot = PlotVideo(video=test_video_path, t=np.arange(100), start_worker=False)
    video_plot._active_plot_videos.add(broken)

    video_plot._cleanup_all_plot_videos()

    assert "Error during close" in capsys.readouterr().out
    assert plot._closed, "a failing plot blocked the others"
    assert not list(video_plot._active_plot_videos)


def test_cleanup_hook_warns_when_workers_outlast_the_drain(capsys):
    """Interpreter shutdown must be bounded, and say so when it gives up."""
    original = video_plot._drain_reaper
    video_plot._drain_reaper = lambda *a, **k: False
    try:
        video_plot._cleanup_all_plot_videos()
    finally:
        video_plot._drain_reaper = original

    assert "Timed out waiting for video workers" in capsys.readouterr().out


def test_drain_is_a_noop_before_any_worker_is_reaped():
    """Draining must not block when the reaper was never started."""
    original = video_plot._reaper_thread
    video_plot._reaper_thread = None
    try:
        assert video_plot._drain_reaper(timeout=0) is True
    finally:
        video_plot._reaper_thread = original
