import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.local.json"

# 所有配置项的默认值
DEFAULTS = {
    # Voyage AI
    "voyage_api_key": None,
    "voyage_model_document": "voyage-4-large",  # Fixed — do not change without re-ingesting
    "voyage_model_query": "voyage-4-large",      # Query embedding model (configurable)
    "embedding_dimensions": 1024,                # Fixed at 1024 to match vector index
    # MongoDB
    "mongodb_uri": None,
    "mongodb_db": "voyage_audio_search",
    "mongodb_collection": "audio_records",
    "search_top_k": 5,
    # STT
    "whisper_model": "mlx-community/whisper-large-v3-mlx",
    # 前端
    "backend_url": "http://localhost:8000",
}


def load_config() -> dict:
    """读取 config.local.json，文件不存在或解析失败时返回空字典。"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    except Exception:
        return {}


def save_config(data: dict) -> None:
    """覆盖写入 config.local.json，只保存非 None 的值。"""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"保存配置失败：{e}") from e


def get_effective_config() -> dict:
    """
    合并策略（优先级从高到低）：
    1. config.local.json 中的非空值
    2. 对应环境变量（仅 voyage_api_key / mongodb_uri 支持）
    3. DEFAULTS 中的默认值
    """
    local = load_config()

    def _resolve(key: str, env_var: str | None = None) -> object:
        val = local.get(key)
        if val is not None and val != "":
            return val
        if env_var:
            env_val = os.environ.get(env_var)
            if env_val:
                return env_val
        return DEFAULTS.get(key)

    return {
        # Voyage AI
        "voyage_api_key":           _resolve("voyage_api_key", "VOYAGE_API_KEY"),
        "voyage_model_document":    _resolve("voyage_model_document"),
        "voyage_model_query":       _resolve("voyage_model_query"),
        "embedding_dimensions":     int(_resolve("embedding_dimensions")),
        # MongoDB
        "mongodb_uri":              _resolve("mongodb_uri", "MONGODB_URI"),
        "mongodb_db":               _resolve("mongodb_db", "MONGODB_DB"),
        "mongodb_collection":       _resolve("mongodb_collection", "MONGODB_COLLECTION"),
        "search_top_k":             int(_resolve("search_top_k")),
        # STT
        "whisper_model":            _resolve("whisper_model"),
        # 前端
        "backend_url":              _resolve("backend_url", "BACKEND_URL"),
    }
