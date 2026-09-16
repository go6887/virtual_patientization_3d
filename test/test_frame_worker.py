"""Verify responsive UI pumping without moving frame state between threads."""

from threading import Event, Thread, get_ident, local

import pytest

from probe_tracking.frame_worker import FrameWorker


def test_pumps_main_thread_while_frame_operation_is_blocked():
    main_thread = get_ident()
    started, release, finished = Event(), Event(), Event()
    pump_threads = []

    def operation(value, *, increment):
        assert get_ident() != main_thread
        started.set()
        try:
            assert release.wait(5), "UI never released the blocked frame"
            return value + increment
        finally:
            finished.set()

    def pump():
        assert started.wait(5), "Frame worker never started"
        pump_threads.append(get_ident())
        if len(pump_threads) <= 3:
            assert not finished.is_set()
        if len(pump_threads) == 3:
            release.set()

    worker = FrameWorker(pump=pump)
    try:
        assert worker.run(operation, 12, increment=7) == 19
        assert len(pump_threads) >= 3
        assert set(pump_threads) == {main_thread}
        assert finished.is_set()
    finally:
        release.set()
        worker.close()


def test_successive_frames_preserve_worker_thread_local_state():
    frame_state = local()
    worker = FrameWorker(pump=lambda: None)

    def first_frame():
        frame_state.timeline = 42
        return get_ident()

    def next_frame():
        return get_ident(), frame_state.timeline

    try:
        thread_id = worker.run(first_frame)
        assert thread_id != get_ident()
        assert worker.run(next_frame) == (thread_id, 42)
    finally:
        worker.close()


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("error", [ValueError("invalid frame"), TimeoutError("camera timed out")])
def test_operation_errors_propagate_and_next_frame_can_run(threaded, error):
    worker = FrameWorker(pump=(lambda: None) if threaded else None)

    def fail():
        raise error

    try:
        with pytest.raises(type(error)) as raised:
            worker.run(fail)
        assert raised.value is error
        assert worker.run(lambda: "next frame") == "next frame"
    finally:
        worker.close()


def test_without_panel_operation_runs_inline_with_arguments():
    worker = FrameWorker()
    main_thread = get_ident()
    try:
        assert worker.run(lambda value, *, label: (get_ident(), value, label), 42, label="frame") == (
            main_thread, 42, "frame"
        )
    finally:
        worker.close()


def test_close_finishes_interrupted_frame_before_resource_cleanup():
    started, release, ready_to_finish, allow_finish = Event(), Event(), Event(), Event()
    close_started, closed = Event(), Event()
    completion_order, close_errors = [], []

    def operation():
        started.set()
        assert release.wait(5), "Test did not release the interrupted frame"
        ready_to_finish.set()
        assert allow_finish.wait(5), "Test did not allow frame cleanup"
        completion_order.append("frame completed")

    def interrupt():
        assert started.wait(5), "Frame worker never started"
        raise KeyboardInterrupt

    worker = FrameWorker(pump=interrupt)

    def cleanup():
        close_started.set()
        try:
            worker.close()
            completion_order.append("resources released")
        except BaseException as error:
            close_errors.append(error)
        finally:
            closed.set()

    closer = Thread(target=cleanup)
    try:
        with pytest.raises(KeyboardInterrupt):
            worker.run(operation)
        closer.start()
        assert close_started.wait(5)
        release.set()
        assert ready_to_finish.wait(5)
        assert not closed.is_set()
        allow_finish.set()
        assert closed.wait(5), "Shutdown did not complete after the frame finished"
        assert not close_errors
        assert completion_order == ["frame completed", "resources released"]
    finally:
        release.set()
        allow_finish.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        worker.close()
