"""
API 端点单元测试
使用 FastAPI TestClient 覆盖主要端点行为
"""
import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """创建 TestClient，mock ping() 使 lifespan 启动成功。"""
    with patch("backend.main.ping", new_callable=AsyncMock, return_value=True):
        from backend.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    """MongoDB 连接正常时返回 200 {"status": "ok"}。"""
    with patch("backend.main.ping", new_callable=AsyncMock, return_value=True):
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_fail(client):
    """MongoDB 连接失败时返回 503。"""
    with patch("backend.main.ping", new_callable=AsyncMock, return_value=False):
        resp = client.get("/health")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "error"


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

def test_ingest_invalid_extension_422(client):
    """上传不支持的格式（.ogg）时返回 422。"""
    file_content = b"fake audio data"
    resp = client.post(
        "/ingest",
        files={"file": ("test.ogg", io.BytesIO(file_content), "audio/ogg")},
    )
    assert resp.status_code == 422
    assert "不支持的文件格式" in resp.json()["detail"]


def test_ingest_success(client):
    """合法音频文件摄入成功，返回 IngestResponse。"""
    with (
        patch("backend.main.transcribe", new_callable=AsyncMock, return_value="测试转录文本"),
        patch("backend.main.embed", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch("backend.main.insert_record", new_callable=AsyncMock, return_value="abc123"),
    ):
        resp = client.post(
            "/ingest",
            files={"file": ("audio.mp3", io.BytesIO(b"fake mp3 data"), "audio/mpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "abc123"
    assert data["filename"] == "audio.mp3"
    assert data["transcript"] == "测试转录文本"


# ---------------------------------------------------------------------------
# POST /search/text
# ---------------------------------------------------------------------------

def test_search_text_empty_query_422(client):
    """空白查询文本返回 422。"""
    resp = client.post("/search/text", json={"query": "   "})
    assert resp.status_code == 422
    assert "查询文本不能为空" in resp.json()["detail"]


def test_search_text_empty_string_422(client):
    """空字符串查询返回 422。"""
    resp = client.post("/search/text", json={"query": ""})
    assert resp.status_code == 422


def test_search_text_empty_db_returns_empty_list(client):
    """空库时搜索返回空列表。"""
    with (
        patch("backend.main.embed", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch("backend.main.vector_search", new_callable=AsyncMock, return_value=[]),
    ):
        resp = client.post("/search/text", json={"query": "搜索关键词"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []


def test_search_text_success(client):
    """文字搜索成功返回结果列表。"""
    mock_results = [
        {"id": "id1", "filename": "a.mp3", "transcript": "内容A", "score": 0.9},
        {"id": "id2", "filename": "b.wav", "transcript": "内容B", "score": 0.8},
    ]
    with (
        patch("backend.main.embed", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch("backend.main.vector_search", new_callable=AsyncMock, return_value=mock_results),
    ):
        resp = client.post("/search/text", json={"query": "搜索关键词"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["filename"] == "a.mp3"


# ---------------------------------------------------------------------------
# POST /search/audio
# ---------------------------------------------------------------------------

def test_search_audio_invalid_extension_422(client):
    """上传不支持格式的音频搜索返回 422。"""
    resp = client.post(
        "/search/audio",
        files={"file": ("query.ogg", io.BytesIO(b"fake"), "audio/ogg")},
    )
    assert resp.status_code == 422


def test_search_audio_empty_db_returns_empty_list(client):
    """空库时音频搜索返回空列表。"""
    with (
        patch("backend.main.transcribe", new_callable=AsyncMock, return_value="查询文本"),
        patch("backend.main.embed", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch("backend.main.vector_search", new_callable=AsyncMock, return_value=[]),
    ):
        resp = client.post(
            "/search/audio",
            files={"file": ("query.mp3", io.BytesIO(b"fake mp3"), "audio/mpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["query_transcript"] == "查询文本"
