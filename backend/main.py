"""
FastAPI 后端主应用
"""
import logging
import sys
import tempfile
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config_service import get_effective_config
from backend.models import (
    IngestResponse,
    SearchResponse,
    SearchTextRequest,
    validate_audio_extension,
    validate_query_text,
)
from backend.services.embedding_service import EmbeddingError, embed
from backend.services.stt_service import STTError, transcribe
from backend.services.vector_store import count_records, insert_record, ping, vector_search
from backend import debug_log

logger = logging.getLogger(__name__)


class IngestPathRequest(BaseModel):
    file_path: str


class ValidateConfigRequest(BaseModel):
    voyage_api_key: str
    voyage_model_document: str
    voyage_model_query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时尝试验证 MongoDB 连接，未配置或连接失败时只警告，不退出
    # （允许先启动服务，再通过 UI 配置 API Key 和 MongoDB URI）
    config = get_effective_config()
    if not config.get("mongodb_uri"):
        logger.warning("MONGODB_URI 未配置，请在 UI 设置中填写后重启服务")
    else:
        ok = await ping()
        if not ok:
            logger.warning("MongoDB 连接失败，请检查 MONGODB_URI 配置")
        else:
            logger.info("MongoDB 连接成功")
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# 全局异常处理器
# ---------------------------------------------------------------------------

@app.exception_handler(STTError)
async def stt_error_handler(request, exc: STTError):
    debug_log.error("API", f"STTError: {exc}")
    return JSONResponse(status_code=500, content={"detail": f"STT 转录失败：{exc}"})


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request, exc: EmbeddingError):
    debug_log.error("API", f"EmbeddingError: {exc}")
    return JSONResponse(status_code=502, content={"detail": f"Embedding 服务失败：{exc}"})


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    ok = await ping()
    if ok:
        return {"status": "ok"}
    return JSONResponse(
        status_code=503,
        content={"status": "error", "detail": "MongoDB 连接失败"},
    )


# ---------------------------------------------------------------------------
# POST /validate-config  （验证 Voyage API Key 和模型是否可用）
# ---------------------------------------------------------------------------

@app.post("/validate-config")
async def validate_config(body: ValidateConfigRequest):
    """
    用传入的 API key 和模型名发一个最小 embed 请求，验证配置是否有效。
    分别验证 document 模型和 query 模型。
    """
    import asyncio
    import voyageai

    results = {}

    def _test_embed(api_key: str, model: str, input_type: str) -> dict:
        try:
            client = voyageai.Client(api_key=api_key)
            client.embed(["test"], model=model, input_type=input_type)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    loop = asyncio.get_event_loop()

    doc_result = await loop.run_in_executor(
        None, lambda: _test_embed(body.voyage_api_key, body.voyage_model_document, "document")
    )
    results["document_model"] = {
        "model": body.voyage_model_document,
        **doc_result,
    }

    # 如果两个模型相同，复用结果
    if body.voyage_model_query == body.voyage_model_document:
        results["query_model"] = {
            "model": body.voyage_model_query,
            **doc_result,
        }
    else:
        query_result = await loop.run_in_executor(
            None, lambda: _test_embed(body.voyage_api_key, body.voyage_model_query, "query")
        )
        results["query_model"] = {
            "model": body.voyage_model_query,
            **query_result,
        }

    all_ok = results["document_model"]["ok"] and results["query_model"]["ok"]
    return {"all_ok": all_ok, "results": results}


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile):
    filename = file.filename or ""
    if not validate_audio_extension(filename):
        ext = os.path.splitext(filename)[-1] or filename
        raise HTTPException(status_code=422, detail=f"不支持的文件格式：{ext}")

    suffix = os.path.splitext(filename)[-1]
    tmp_path = None
    debug_log.info("API", "POST /ingest", {"filename": filename})
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        transcript = await transcribe(tmp_path)
        embedding = await embed(transcript, input_type="document")
        record_id = await insert_record(filename, transcript, embedding)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    debug_log.info("API", "POST /ingest 完成", {"record_id": record_id, "filename": filename})
    return IngestResponse(id=record_id, filename=filename, transcript=transcript)


# ---------------------------------------------------------------------------
# POST /ingest/path  （通过本地文件路径摄入，filename 存完整路径）
# ---------------------------------------------------------------------------

@app.post("/ingest/path", response_model=IngestResponse)
async def ingest_path(body: IngestPathRequest):
    file_path = body.file_path.strip()
    if not file_path:
        raise HTTPException(status_code=422, detail="文件路径不能为空")
    if not validate_audio_extension(file_path):
        ext = os.path.splitext(file_path)[-1] or file_path
        raise HTTPException(status_code=422, detail=f"不支持的文件格式：{ext}")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=422, detail=f"文件不存在：{file_path}")

    debug_log.info("API", "POST /ingest/path", {"file_path": file_path})
    transcript = await transcribe(file_path)
    embedding = await embed(transcript, input_type="document")
    record_id = await insert_record(file_path, transcript, embedding)
    debug_log.info("API", "POST /ingest/path 完成", {"record_id": record_id})
    return IngestResponse(id=record_id, filename=file_path, transcript=transcript)


# ---------------------------------------------------------------------------
# POST /search/audio
# ---------------------------------------------------------------------------

@app.post("/search/audio", response_model=SearchResponse)
async def search_audio(file: UploadFile):
    filename = file.filename or ""
    if not validate_audio_extension(filename):
        ext = os.path.splitext(filename)[-1] or filename
        raise HTTPException(status_code=422, detail=f"不支持的文件格式：{ext}")

    suffix = os.path.splitext(filename)[-1]
    tmp_path = None
    debug_log.info("API", "POST /search/audio", {"filename": filename})
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        transcript = await transcribe(tmp_path)
        embedding = await embed(transcript, input_type="query")
        results = await vector_search(embedding)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    debug_log.info("API", "POST /search/audio 完成", {"results_count": len(results)})
    return SearchResponse(query_transcript=transcript, results=results)


@app.post("/search/text", response_model=SearchResponse)
async def search_text(body: SearchTextRequest):
    if not validate_query_text(body.query):
        raise HTTPException(status_code=422, detail="查询文本不能为空")

    debug_log.info("API", "POST /search/text", {
        "query_preview": body.query[:80] + ("..." if len(body.query) > 80 else ""),
    })
    embedding = await embed(body.query, input_type="query")
    results = await vector_search(embedding)
    debug_log.info("API", "POST /search/text 完成", {"results_count": len(results)})
    return SearchResponse(query_transcript=None, results=results)


# ---------------------------------------------------------------------------
# GET /debug/logs  （返回最近的 debug 日志）
# GET /debug/clear （清空日志）
# ---------------------------------------------------------------------------

@app.get("/debug/logs")
async def get_debug_logs(limit: int = 100):
    return {"logs": debug_log.get_logs(limit=limit)}


@app.post("/debug/clear")
async def clear_debug_logs():
    debug_log.clear_logs()
    return {"status": "cleared"}
