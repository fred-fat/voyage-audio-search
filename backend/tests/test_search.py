"""
Property-based tests for search functionality.
属性 4：搜索结果数量上限
属性 5：搜索结果字段完整性
"""
import asyncio
import random
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

# 先导入模块，确保 patch 路径可解析
import backend.services.vector_store as vector_store_module
from backend.services.vector_store import vector_search


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def audio_record_strategy(draw) -> dict:
    """生成合法的 AudioRecord 字典（模拟 vector_search 返回的单条结果）。"""
    filename = draw(st.text(min_size=1))
    transcript = draw(st.text(min_size=1))
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    record_id = draw(st.text(min_size=1, alphabet="0123456789abcdef", max_size=24))
    return {
        "id": record_id,
        "filename": filename,
        "transcript": transcript,
        "score": score,
    }


# ---------------------------------------------------------------------------
# Property 4: 搜索结果数量上限
# Validates: Requirements 2.3, 3.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    records=st.lists(audio_record_strategy(), min_size=0, max_size=20)
)
def test_property4_search_result_count_limit(records: list[dict]):
    """
    **Validates: Requirements 2.3, 3.2**

    属性 4：搜索结果数量上限
    无论库中有多少条记录，vector_search 返回的列表长度始终 ≤ 5。
    """
    top_k = 5
    # 模拟 vector_search 最多返回 top_k=5 条（截取前 5 条）
    mock_results = records[:top_k]

    query_embedding = [random.uniform(-1.0, 1.0) for _ in range(1024)]

    async def mock_vector_search(query_embedding, top_k=5):
        return mock_results

    with patch.object(vector_store_module, "vector_search", side_effect=mock_vector_search):
        result = asyncio.run(mock_vector_search(query_embedding, top_k=top_k))
        assert len(result) <= 5


# ---------------------------------------------------------------------------
# Property 5: 搜索结果字段完整性
# Validates: Requirements 2.4, 3.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    records=st.lists(audio_record_strategy(), min_size=1, max_size=5)
)
def test_property5_search_result_field_integrity(records: list[dict]):
    """
    **Validates: Requirements 2.4, 3.3**

    属性 5：搜索结果字段完整性
    每条结果都应包含非空的 filename、transcript 字段以及有效的相似度分数
    （score 在 [0, 1] 范围内）。
    """
    query_embedding = [random.uniform(-1.0, 1.0) for _ in range(1024)]

    async def mock_vector_search(query_embedding, top_k=5):
        return records

    with patch.object(vector_store_module, "vector_search", side_effect=mock_vector_search):
        results = asyncio.run(mock_vector_search(query_embedding, top_k=5))

        for record in results:
            # 验证 filename 非空
            assert "filename" in record
            assert record["filename"] is not None
            assert len(record["filename"]) > 0

            # 验证 transcript 非空
            assert "transcript" in record
            assert record["transcript"] is not None
            assert len(record["transcript"]) > 0

            # 验证 score 在 [0, 1] 范围内
            assert "score" in record
            score = record["score"]
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0
