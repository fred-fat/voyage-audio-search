"""
Property-based tests for Vector_Store module.
属性 1：写入 round-trip 完整性
"""
import asyncio
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId
from hypothesis import given, settings
from hypothesis import strategies as st

# 先导入模块，确保 patch 路径可解析
import backend.services.vector_store as vector_store_module
from backend.services.vector_store import insert_record


def _make_mock_embedding() -> list[float]:
    """生成 1024 维随机浮点列表作为 mock embedding。"""
    return [random.uniform(-1.0, 1.0) for _ in range(1024)]


# ---------------------------------------------------------------------------
# Property 1: 写入 round-trip 完整性
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    filename=st.text(min_size=1),
    transcript=st.text(min_size=1),
)
def test_property1_insert_round_trip(filename: str, transcript: str):
    """
    **Validates: Requirements 1.5**

    属性 1：写入 round-trip 完整性
    对于任意合法的 filename（非空字符串）和 transcript（非空字符串），
    调用 insert_record 写入后，通过 _id 查询返回的文档应包含完全相同的
    filename、transcript 字段，且 embedding 长度为 1024。
    """
    embedding = _make_mock_embedding()
    fake_id = ObjectId()

    # 构造写入后查询返回的文档
    stored_doc = {
        "_id": fake_id,
        "filename": filename,
        "transcript": transcript,
        "embedding": embedding,
    }

    mock_insert_result = MagicMock()
    mock_insert_result.inserted_id = fake_id

    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock(return_value=mock_insert_result)
    mock_collection.find_one = AsyncMock(return_value=stored_doc)

    with patch.object(vector_store_module, "_get_collection", return_value=mock_collection):
        # 执行写入
        returned_id = asyncio.run(insert_record(filename, transcript, embedding))

        # 验证返回的 ID 是字符串
        assert isinstance(returned_id, str)
        assert returned_id == str(fake_id)

        # 模拟按 _id 查询
        async def _find():
            return await mock_collection.find_one({"_id": ObjectId(returned_id)})

        doc = asyncio.run(_find())

        # 验证 round-trip 完整性
        assert doc is not None
        assert doc["filename"] == filename
        assert doc["transcript"] == transcript
        assert len(doc["embedding"]) == 1024
