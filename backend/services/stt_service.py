import asyncio
import time

try:
    import mlx_whisper
except ImportError:  # pragma: no cover – only available on Apple Silicon
    mlx_whisper = None  # type: ignore

from backend import debug_log


class STTError(Exception):
    """STT 转录失败时抛出。"""
    pass


async def transcribe(file_path: str) -> str:
    """
    将音频文件转录为文本。
    使用 asyncio.run_in_executor 包装同步的 mlx_whisper.transcribe()。
    """
    from backend.config_service import get_effective_config
    model = get_effective_config().get("whisper_model", "mlx-community/whisper-large-v3-mlx")

    debug_log.info("STT", "开始转录", {
        "file": file_path,
        "model": model,
    })

    t0 = time.time()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: mlx_whisper.transcribe(file_path, path_or_hf_repo=model),
        )
        text = result["text"].strip()
        elapsed = round(time.time() - t0, 2)
        debug_log.info("STT", "转录完成", {
            "model": model,
            "elapsed_s": elapsed,
            "transcript_preview": text[:100] + ("..." if len(text) > 100 else ""),
            "transcript_length": len(text),
        })
        return text
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        debug_log.error("STT", f"转录失败: {e}", {
            "model": model,
            "file": file_path,
            "elapsed_s": elapsed,
        })
        raise STTError(f"转录失败: {e}") from e
