"""
Vector_Store 模块：封装 MongoDB Atlas 的异步操作。
使用 motor（异步 MongoDB 驱动）实现连接管理和向量搜索。
所有配置（URI、DB、集合名、top_k）均从 get_effective_config() 动态读取。
"""
import time
from typing import Optional

import motor.motor_asyncio
from bson import ObjectId

from backend.config_service import get_effective_config
from backend import debug_log

# 模块级客户端（懒加载，URI 变更时需重置）
_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
_last_uri: Optional[str] = None


def _get_collection() -> motor.motor_asyncio.AsyncIOMotorCollection:
    """获取 motor 集合对象（懒加载，URI 变更时自动重建客户端）。"""
    global _client, _last_uri
    config = get_effective_config()
    mongodb_uri = config.get("mongodb_uri")
    if not mongodb_uri:
        raise RuntimeError("MongoDB URI 未配置，请在设置中填写 MONGODB_URI 或设置环境变量。")
    if _client is None or _last_uri != mongodb_uri:
        _client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_uri)
        _last_uri = mongodb_uri
    db_name = config.get("mongodb_db", "voyage_audio_search")
    collection_name = config.get("mongodb_collection", "audio_records")
    return _client[db_name][collection_name]


async def insert_record(filename: str, transcript: str, embedding: list[float]) -> str:
    """写入 Audio_Record，返回 ObjectId 字符串。"""
    from datetime import datetime, timezone
    config = get_effective_config()
    collection = _get_collection()

    debug_log.info("MONGO", "insert_record", {
        "db": config.get("mongodb_db"),
        "collection": config.get("mongodb_collection"),
        "filename": filename,
        "transcript_preview": transcript[:80] + ("..." if len(transcript) > 80 else ""),
        "embedding_dims": len(embedding),
    })

    t0 = time.time()
    doc = {
        "filename": filename,
        "transcript": transcript,
        "embedding": embedding,
        "created_at": datetime.now(timezone.utc),
    }
    result = await collection.insert_one(doc)
    record_id = str(result.inserted_id)
    elapsed = round(time.time() - t0, 2)

    debug_log.info("MONGO", "insert_record 完成", {
        "inserted_id": record_id,
        "elapsed_s": elapsed,
    })
    return record_id


async def vector_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    """
    执行 $vectorSearch 聚合，返回 Top-K 结果（含 score 字段）。
    top_k 默认从 get_effective_config()["search_top_k"] 读取。
    """
    config = get_effective_config()
    if top_k is None:
        top_k = int(config.get("search_top_k", 5))

    db_name = config.get("mongodb_db")
    col_name = config.get("mongodb_collection")

    debug_log.info("MONGO", "$vectorSearch", {
        "db": db_name,
        "collection": col_name,
        "index": "embedding_index",
        "top_k": top_k,
        "num_candidates": top_k * 10,
        "query_embedding_dims": len(query_embedding),
        "query_vector_preview": [round(v, 4) for v in query_embedding[:5]],
    })

    collection = _get_collection()
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embedding_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": top_k * 10,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 1,
                "filename": 1,
                "transcript": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    t0 = time.time()
    cursor = collection.aggregate(pipeline)
    results = []
    async for doc in cursor:
        results.append({
            "id": str(doc["_id"]),
            "filename": doc["filename"],
            "transcript": doc["transcript"],
            "score": doc["score"],
        })
    elapsed = round(time.time() - t0, 2)

    debug_log.info("MONGO", "$vectorSearch 完成", {
        "results_count": len(results),
        "elapsed_s": elapsed,
        "scores": [round(r["score"], 4) for r in results],
        "filenames": [r["filename"] for r in results],
    })
    return results


async def count_records() -> int:
    """返回集合中的文档总数。"""
    collection = _get_collection()
    return await collection.count_documents({})


async def ping() -> bool:
    """ping MongoDB，返回 True/False。"""
    try:
        config = get_effective_config()
        mongodb_uri = config.get("mongodb_uri")
        if not mongodb_uri:
            return False
        global _client, _last_uri
        if _client is None or _last_uri != mongodb_uri:
            _client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_uri)
            _last_uri = mongodb_uri
        await _client.admin.command("ping")
        return True
    except Exception:
        return False
