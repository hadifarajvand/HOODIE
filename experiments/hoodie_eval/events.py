from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from environment import Environment


class EventRecorder:
    """Collects task-level events during simulation for post-hoc analysis."""

    def __init__(self) -> None:
        self.events: list[Dict[str, Any]] = []
        self._next_task_id = 0
        self._task_metadata: Dict[int, Dict[str, Any]] = {}

    def _ensure_task_id(self, task: Any) -> int:
        if not hasattr(task, "_hoodie_id"):
            setattr(task, "_hoodie_id", self._next_task_id)
            self._next_task_id += 1
        return int(getattr(task, "_hoodie_id"))

    def record_arrivals(
        self, time_step: int, tasks: Sequence[Any], server_ids: Iterable[int]
    ) -> None:
        for server_id, task in zip(server_ids, tasks):
            if task is None:
                continue
            task_id = self._ensure_task_id(task)
            self._task_metadata[task_id] = {
                "arrival_time": getattr(task, "arrival_time", time_step),
                "origin": getattr(task, "get_origin_server_id", lambda: server_id)(),
            }
            event = {
                "event": "task_arrived",
                "time": time_step,
                "server": server_id,
                "task_id": task_id,
                "arrival_time": getattr(task, "arrival_time", time_step),
                "timeout": getattr(task, "timeout_delay", None),
                "size": getattr(task, "size", None),
                "priority": getattr(task, "priotiry", None),
            }
            self.events.append(event)

    def record_actions(self, time_step: int, actions: Sequence[int]) -> None:
        self.events.append(
            {
                "event": "actions",
                "time": time_step,
                "load": sum(action is not None for action in actions) / max(len(actions), 1),
                "actions": list(actions),
            }
        )

    def record_info(self, time_step: int, info: Mapping[str, Any]) -> None:
        self.events.append(
            {
                "event": "info",
                "time": time_step,
                "payload": dict(info),
            }
        )

    def record_queue_lengths(self, env: Environment, time_step: int) -> None:
        for server in env.servers:
            self.events.append(
                {
                    "event": "queue_length",
                    "time": time_step,
                    "queue": f"server_{server.id}_private",
                    "length": getattr(server.processing_queue, "queue_length", 0.0),
                }
            )
            self.events.append(
                {
                    "event": "queue_length",
                    "time": time_step,
                    "queue": f"server_{server.id}_offloading",
                    "length": getattr(server.offloading_queue, "queue_length", 0.0),
                }
            )
            for origin_id, public_queue in server.public_queue_manager.public_queues.items():
                self.events.append(
                    {
                        "event": "queue_length",
                        "time": time_step,
                        "queue": f"server_{server.id}_public_from_{origin_id}",
                        "length": getattr(public_queue, "queue_length", 0.0),
                    }
                )
        cloud_manager = env.cloud.public_queue_manager
        for origin_id, public_queue in cloud_manager.public_queues.items():
            self.events.append(
                {
                    "event": "queue_length",
                    "time": time_step,
                    "queue": f"cloud_public_from_{origin_id}",
                    "length": getattr(public_queue, "queue_length", 0.0),
                }
            )

    def record_completion(self, task: Any, finish_time: float, context: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
        task_id = self._ensure_task_id(task)
        metadata = self._task_metadata.get(task_id, {})
        arrival_time = metadata.get("arrival_time", snapshot.get("arrival_time", finish_time))
        latency = float(finish_time - arrival_time)
        event = {
            "event": "task_completed",
            "task_id": task_id,
            "time": finish_time,
            "latency": latency,
            "origin": snapshot.get("origin", metadata.get("origin")),
            "target": snapshot.get("target"),
            "destination": "cloud" if context.get("executor_type") == "cloud" else "edge",
            "executor": context.get("server_id"),
            "queue_type": context.get("queue_type"),
        }
        self.events.append(event)
        self._task_metadata.pop(task_id, None)

    def record_drop(self, task: Any, penalty: float, context: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
        task_id = self._ensure_task_id(task)
        queue = context.get("queue")
        drop_time = getattr(queue, "current_time", None)
        event = {
            "event": "task_dropped",
            "task_id": task_id,
            "time": drop_time,
            "penalty": penalty,
            "origin": snapshot.get("origin"),
            "target": snapshot.get("target"),
            "destination": "cloud" if context.get("queue_type") == "cloud_public" else "edge",
            "queue_type": context.get("queue_type"),
        }
        self.events.append(event)
        self._task_metadata.pop(task_id, None)
