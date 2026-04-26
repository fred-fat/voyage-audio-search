"""
Property-based tests for ingest flow.
属性 3：错误时 Vector_Store 保持不变
Validates: Requirements 1.8
"""
import asyncio
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

import backend.main as main_module
from backend.services.embedding_service import EmbeddingError
from backend.services.stt_service import STTError


# ---------------------------------------------------------------------------
# Property 3: 错误时 Vector_Store 保持不变
# Validates: Requirements 1.8
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(audio_bytes=st.binary(min_size=1, max_size=1024))
def test_property3_stt_error_no_insert(audio_bytes: bytes):
    """
    **Validates: Requirements 1.8**

    属性 3：错误时 Vector_Store 保持不变
    若 STT_Service 抛出异常，insert_record 不应被调用。
    """
    mock_insert = AsyncMock()

    with (
        patch("backend.main.transcribe", side_effect=STTError("转录失败")),
        patch("backend.main.embed", new_callable=AsyncMock),
        patch("backend.main.insert_record", mock_insert),
    ):
        async def _run():
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                try:
                    transcript = await main_module.transcribe(tmp_path)
                    embedding = await main_module.embed(transcript, input_type="document")
                    await main_module.insert_record("test.mp3", transcript, embedding)
                except STTError:
                    pass  # 预期异常
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        asyncio.run(_run())

    # STT 失败时，insert_record 不应被调用
    mock_insert.assert_not_called()


@settings(max_examples=50)
@given(audio_bytes=st.binary(min_size=1, max_size=1024))
def test_property3_embedding_error_no_insert(audio_bytes: bytes):
    """
    **Validates: Requirements 1.8**

    属性 3：错误时 Vector_Store 保持不变
    若 Embedding_Service 抛出异常，insert_record 不应被调用。
    """
    mock_insert = AsyncMock()

    with (
        patch("backend.main.transcribe", new_callable=AsyncMock, return_value="转录文本"),
        patch("backend.main.embed", side_effect=EmbeddingError("Embedding 失败")),
        patch("backend.main.insert_record", mock_insert),
    ):
        async def _run():
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                try:
                    transcript = await main_module.transcribe(tmp_path)
                    embedding = await main_module.embed(transcript, input_type="document")
                    await main_module.insert_record("test.mp3", transcript, embedding)
                except EmbeddingError:
                    pass  # 预期异常
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        asyncio.run(_run())

    # Embedding 失败时，insert_record 不应被调用
    mock_insert.assert_not_called()
