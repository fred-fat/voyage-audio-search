import asyncio
import time
import voyageai

from backend.config_service import get_effective_config
from backend import debug_log


class EmbeddingError(Exception):
    """Embedding 调用失败时抛出。"""
    pass


async def embed(text: str, input_type: str = "document") -> list[float]:
    """
    将文本转换为向量。
    - input_type="document" 时使用 voyage_model_document
    - input_type="query"    时使用 voyage_model_query
    """
    config = get_effective_config()
    api_key = config["voyage_api_key"]
    if input_type == "query":
        model = config.get("voyage_model_query", "voyage-3")
    else:
        model = config.get("voyage_model_document", "voyage-3")

    debug_log.info("EMBED", f"开始 embedding [{input_type}]", {
        "model": model,
        "input_type": input_type,
        "text_preview": text[:100] + ("..." if len(text) > 100 else ""),
        "text_length": len(text),
    })

    t0 = time.time()

    def _call():
        client = voyageai.Client(api_key=api_key)
        result = client.embed([text], model=model, input_type=input_type)
        return result.embeddings[0]

    try:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, _call)
        elapsed = round(time.time() - t0, 2)
        debug_log.info("EMBED", f"Embedding 完成 [{input_type}]", {
            "model": model,
            "input_type": input_type,
            "dimensions": len(embedding),
            "elapsed_s": elapsed,
        })
        return embedding
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        debug_log.error("EMBED", f"Embedding 失败: {e}", {
            "model": model,
            "input_type": input_type,
            "elapsed_s": elapsed,
        })
        raise EmbeddingError(f"Embedding 调用失败：{e}") from e
