import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

import backend.config_service as cs


@pytest.fixture(autouse=True)
def isolate_config_path(tmp_path, monkeypatch):
    """将 CONFIG_PATH 重定向到临时目录，避免污染真实文件。"""
    fake_path = tmp_path / "config.local.json"
    monkeypatch.setattr(cs, "CONFIG_PATH", fake_path)
    yield fake_path


# --- load_config ---

def test_load_config_returns_empty_dict_when_file_missing():
    assert cs.load_config() == {}


def test_load_config_returns_data_when_file_exists(isolate_config_path):
    isolate_config_path.write_text(
        json.dumps({"voyage_api_key": "vk-123", "mongodb_uri": "mongodb://localhost"})
    )
    result = cs.load_config()
    assert result["voyage_api_key"] == "vk-123"
    assert result["mongodb_uri"] == "mongodb://localhost"


def test_load_config_returns_empty_dict_on_invalid_json(isolate_config_path):
    isolate_config_path.write_text("not valid json")
    assert cs.load_config() == {}


# --- save_config ---

def test_save_config_writes_file(isolate_config_path):
    cs.save_config({"voyage_api_key": "key-abc", "mongodb_uri": "mongodb://host"})
    data = json.loads(isolate_config_path.read_text())
    assert data["voyage_api_key"] == "key-abc"
    assert data["mongodb_uri"] == "mongodb://host"


def test_save_config_overwrites_existing(isolate_config_path):
    cs.save_config({"voyage_api_key": "old-key", "mongodb_uri": "old-uri"})
    cs.save_config({"voyage_api_key": "new-key", "mongodb_uri": "new-uri"})
    data = json.loads(isolate_config_path.read_text())
    assert data["voyage_api_key"] == "new-key"
    assert data["mongodb_uri"] == "new-uri"


# --- get_effective_config ---

def test_get_effective_config_uses_local_file(isolate_config_path):
    isolate_config_path.write_text(
        json.dumps({"voyage_api_key": "local-key", "mongodb_uri": "local-uri"})
    )
    with patch.dict(os.environ, {}, clear=True):
        result = cs.get_effective_config()
    assert result["voyage_api_key"] == "local-key"
    assert result["mongodb_uri"] == "local-uri"


def test_get_effective_config_falls_back_to_env(isolate_config_path):
    # 文件不存在，fallback 到环境变量
    env = {"VOYAGE_API_KEY": "env-key", "MONGODB_URI": "env-uri"}
    with patch.dict(os.environ, env, clear=True):
        result = cs.get_effective_config()
    assert result["voyage_api_key"] == "env-key"
    assert result["mongodb_uri"] == "env-uri"


def test_get_effective_config_empty_string_falls_back_to_env(isolate_config_path):
    # 空字符串视为未配置
    isolate_config_path.write_text(
        json.dumps({"voyage_api_key": "", "mongodb_uri": ""})
    )
    env = {"VOYAGE_API_KEY": "env-key", "MONGODB_URI": "env-uri"}
    with patch.dict(os.environ, env, clear=True):
        result = cs.get_effective_config()
    assert result["voyage_api_key"] == "env-key"
    assert result["mongodb_uri"] == "env-uri"


def test_get_effective_config_returns_none_when_nothing_configured(isolate_config_path):
    with patch.dict(os.environ, {}, clear=True):
        result = cs.get_effective_config()
    assert result["voyage_api_key"] is None
    assert result["mongodb_uri"] is None


def test_get_effective_config_local_overrides_env(isolate_config_path):
    isolate_config_path.write_text(
        json.dumps({"voyage_api_key": "local-key", "mongodb_uri": "local-uri"})
    )
    env = {"VOYAGE_API_KEY": "env-key", "MONGODB_URI": "env-uri"}
    with patch.dict(os.environ, env, clear=True):
        result = cs.get_effective_config()
    assert result["voyage_api_key"] == "local-key"
    assert result["mongodb_uri"] == "local-uri"
