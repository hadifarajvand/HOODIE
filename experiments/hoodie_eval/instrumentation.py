from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from environment.queues import OffloadingQueue, ProcessingQueue, PublicQueue, TaskQueue
from environment.task import Task

if TYPE_CHECKING:  # pragma: no cover
    from environment.environment import Environment
    from .runner import EventRecorder

_CONTEXT_STACK: List[Dict[str, Any]] = []
_RECORDER: Optional["EventRecorder"] = None
_PATCHED = False


def _push_context(context: Optional[Dict[str, Any]]) -> None:
    if context is not None:
        _CONTEXT_STACK.append(context)


def _pop_context() -> None:
    if _CONTEXT_STACK:
        _CONTEXT_STACK.pop()


def _current_context() -> Optional[Dict[str, Any]]:
    if not _CONTEXT_STACK:
        return None
    return _CONTEXT_STACK[-1]


def _bind_recorder(recorder: "EventRecorder") -> None:
    global _RECORDER
    _RECORDER = recorder


def _task_snapshot(task: Task) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "arrival_time": getattr(task, "arrival_time", None),
        "timeout": getattr(task, "timeout_delay", None),
    }
    for key, attr in ("origin", "get_origin_server_id"), ("target", "get_target_server_id"):
        getter = getattr(task, attr, None)
        if callable(getter):
            try:
                snapshot[key] = getter()
            except AssertionError:
                snapshot[key] = None
            except Exception:
                snapshot[key] = None
        else:
            snapshot[key] = getattr(task, key, None)
    return snapshot


def _log_completion(task: Task, finish_time: float, snapshot: Dict[str, Any]) -> None:
    if _RECORDER is None:
        return
    context = _current_context()
    if context is None:
        return
    _RECORDER.record_completion(task, finish_time, context, snapshot)


def _log_drop(task: Task, penalty: float, snapshot: Dict[str, Any]) -> None:
    if _RECORDER is None:
        return
    context = _current_context()
    if context is None:
        return
    _RECORDER.record_drop(task, penalty, context, snapshot)


def _ensure_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    original_processing_step = ProcessingQueue.step
    original_offloading_step = OffloadingQueue.step
    original_public_step = PublicQueue.step
    original_get_first = TaskQueue.get_first_non_empty_element
    original_finish_task = Task.finish_task
    original_drop_task = Task.drop_task
    original_copy = Task.copy
    original_transmit = Task.transmit

    def processing_step(self: ProcessingQueue):
        _push_context(getattr(self, "_hoodie_context", None))
        try:
            return original_processing_step(self)
        finally:
            _pop_context()

    def offloading_step(self: OffloadingQueue):
        _push_context(getattr(self, "_hoodie_context", None))
        try:
            return original_offloading_step(self)
        finally:
            _pop_context()

    def public_step(self: PublicQueue, computational_capacity: float):
        _push_context(getattr(self, "_hoodie_context", None))
        try:
            return original_public_step(self, computational_capacity)
        finally:
            _pop_context()

    def get_first_non_empty(self: TaskQueue):
        _push_context(getattr(self, "_hoodie_context", None))
        try:
            return original_get_first(self)
        finally:
            _pop_context()

    def finish_task(self: Task, finish_time: int):
        snapshot = _task_snapshot(self)
        result = original_finish_task(self, finish_time)
        _log_completion(self, finish_time, snapshot)
        return result

    def drop_task(self: Task) -> int:
        snapshot = _task_snapshot(self)
        penalty = original_drop_task(self)
        _log_drop(self, penalty, snapshot)
        return penalty

    def task_copy(self: Task):
        new_task = original_copy(self)
        if hasattr(self, "_hoodie_id"):
            setattr(new_task, "_hoodie_id", getattr(self, "_hoodie_id"))
        return new_task

    def task_transmit(self: Task, offloading_capacity: float):
        new_task = original_transmit(self, offloading_capacity)
        if new_task is not None and hasattr(self, "_hoodie_id"):
            setattr(new_task, "_hoodie_id", getattr(self, "_hoodie_id"))
        return new_task

    ProcessingQueue.step = processing_step  # type: ignore[assignment]
    OffloadingQueue.step = offloading_step  # type: ignore[assignment]
    PublicQueue.step = public_step  # type: ignore[assignment]
    TaskQueue.get_first_non_empty_element = get_first_non_empty  # type: ignore[assignment]
    Task.finish_task = finish_task  # type: ignore[assignment]
    Task.drop_task = drop_task  # type: ignore[assignment]
    Task.copy = task_copy  # type: ignore[assignment]
    Task.transmit = task_transmit  # type: ignore[assignment]

    _PATCHED = True


def _bind_queue_contexts(env: "Environment") -> None:
    for server in env.servers:
        server.processing_queue._hoodie_context = {
            "queue_type": "private",
            "executor_type": "edge",
            "server_id": server.id,
            "queue": server.processing_queue,
        }
        server.offloading_queue._hoodie_context = {
            "queue_type": "offloading",
            "executor_type": "edge",
            "server_id": server.id,
            "queue": server.offloading_queue,
        }
        for origin_id, public_queue in server.public_queue_manager.public_queues.items():
            public_queue._hoodie_context = {
                "queue_type": "public",
                "executor_type": "edge",
                "server_id": server.id,
                "origin_id": origin_id,
                "queue": public_queue,
            }
    cloud = env.cloud
    for origin_id, public_queue in cloud.public_queue_manager.public_queues.items():
        public_queue._hoodie_context = {
            "queue_type": "cloud_public",
            "executor_type": "cloud",
            "server_id": getattr(cloud, "number_of_servers", None),
            "origin_id": origin_id,
            "queue": public_queue,
        }


def attach_instrumentation(env: "Environment", recorder: "EventRecorder") -> None:
    """
    Patch queue/task methods (once) and bind the current recorder so that task
    completions and drops generate structured events.
    """

    _ensure_patches()
    _bind_recorder(recorder)
    _bind_queue_contexts(env)
