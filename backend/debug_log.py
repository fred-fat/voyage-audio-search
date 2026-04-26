"""
Debug Log 模块：内存 RingBuffer，记录每次操作的详细执行信息。
最多保留 200 条，后端重启后清空。
"""
import time
from collections import deque
from datetime import datetime
from typing import Any

_MAX_ENTRIES = 200
_log_buffer: deque = deque(maxlen=_MAX_ENTRIES)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(level: str, category: str, message: str, detail: dict[str, Any] | None = None) -> None:
    """写入一条 debug 日志。"""
    entry = {
        "ts": _ts(),
        "level": level,       # INFO / DEBUG / ERROR
        "category": category, # STT / EMBED / MONGO / API
        "message": message,
        "detail": detail or {},
    }
    _log_buffer.append(entry)


def info(category: str, message: str, detail: dict[str, Any] | None = None) -> None:
    log("INFO", category, message, detail)


def debug(category: str, message: str, detail: dict[str, Any] | None = None) -> None:
    log("DEBUG", category, message, detail)


def error(category: str, message: str, detail: dict[str, Any] | None = None) -> None:
    log("ERROR", category, message, detail)


def get_logs(limit: int = 100) -> list[dict]:
    """返回最近 limit 条日志（最新的在前）。"""
    entries = list(_log_buffer)
    return list(reversed(entries))[:limit]


def clear_logs() -> None:
    _log_buffer.clear()
