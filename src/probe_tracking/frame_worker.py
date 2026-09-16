"""Run frame work serially while the main thread services desktop controls."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait


class FrameWorker:
    """Keep Tk/OpenCV windows on the main thread and frame state on one worker.

    Only one operation is submitted at a time: camera reads cannot build up a
    backlog, and Rerun's thread-local timelines stay on the same worker. Without
    a panel, execute inline to preserve the headless tracking path.
    """

    def __init__(self, pump: Callable[[], None] | None = None) -> None:
        self._pump = pump
        self._executor = None if pump is None else ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracking")

    def run(self, operation, /, *args, **kwargs):
        if self._executor is None:
            return operation(*args, **kwargs)
        future = self._executor.submit(operation, *args, **kwargs)
        while not future.done():
            self._pump()
            wait((future,), timeout=0.008)
        return future.result()

    def close(self) -> None:
        # Finish in-flight work before releasing the capture or recording sink,
        # including when Ctrl+C interrupts the main thread during run().
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
