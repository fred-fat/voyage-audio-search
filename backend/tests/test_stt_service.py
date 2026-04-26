import sys
import types
import pytest
from unittest.mock import patch, MagicMock

# mlx_whisper 仅在 Apple Silicon 上可用，测试时用 stub 替代
_stub = types.ModuleType("mlx_whisper")
_stub.transcribe = MagicMock()  # type: ignore
sys.modules["mlx_whisper"] = _stub

# 重新导入，确保模块使用 stub
import importlib
import backend.services.stt_service as _stt_mod
_stt_mod.mlx_whisper = _stub  # type: ignore
importlib.reload(_stt_mod)

from backend.services.stt_service import transcribe, STTError


@pytest.mark.asyncio
async def test_transcribe_returns_string():
    """正常转录返回字符串。"""
    mock_result = {"text": "Hello world"}
    with patch.object(_stub, "transcribe", return_value=mock_result):
        result = await transcribe("audio.mp3")
    assert result == "Hello world"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_transcribe_raises_stt_error_on_exception():
    """mlx_whisper 抛出异常时，STTError 被正确传播。"""
    with patch.object(_stub, "transcribe", side_effect=RuntimeError("model error")):
        with pytest.raises(STTError):
            await transcribe("audio.mp3")


@pytest.mark.asyncio
async def test_transcribe_strips_whitespace():
    """返回值是 strip 后的字符串（去除首尾空白）。"""
    mock_result = {"text": "  transcribed text  \n"}
    with patch.object(_stub, "transcribe", return_value=mock_result):
        result = await transcribe("audio.mp3")
    assert result == "transcribed text"
