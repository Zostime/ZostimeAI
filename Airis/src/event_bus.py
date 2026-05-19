from typing import Any, Literal
import threading
import queue
import time

from . import runtime

class EventBus:
    PRIORITY_MAP = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }  # EVENT 优先级

    def __init__(self):
        self._event_queue = queue.PriorityQueue()
        threading.Thread(target=self._loop, daemon=True).start()
        self._handlers = {}
        self._lock = threading.Lock()

    def on(self, event_type: str, handler) -> None:
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def emit(self,
             event_type: str,
             data: Any = None,
             priority: Literal["low", "medium", "high", "critical"] = "low"
             ) -> None:
        if priority not in self.PRIORITY_MAP:
            raise ValueError(f"未定义的EVENT优先级: {priority}")

        priority_val = self.PRIORITY_MAP[priority]
        timestamp = time.time()
        event = {
            "timestamp": timestamp,
            "priority": priority,
            "type": event_type,
            "data": data
        }
        # (优先级, 时间戳, event)
        self._event_queue.put((priority_val, timestamp, event))

    def _loop(self):
        while True:
            _, _, event = self._event_queue.get()
            if event is None:
                break

            with self._lock:
                handlers = list(self._handlers.get(event["type"], [])) + \
                           list(self._handlers.get("*", []))

            for handler in handlers:
                try:
                    threading.Thread(
                        target=handler,
                        args=(event,),
                        daemon=True
                    ).start()
                except Exception as e:
                    runtime.LOGGER.logger.error(f"[Event Error] {event['type']} -> {e}")
