import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_voyage_client():
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1] * 1024]
    mock_client = MagicMock()
    mock_client.embed.return_value = mock_result
    with patch("backend.services.embedding_service.voyageai.Client", return_value=mock_client) as mock_cls:
        yield mock_cls, mock_client


@pytest.mark.asyncio
async def test_embed_returns_1024_floats(mock_voyage_client):
    from backend.services.embedding_service import embed
    result = await embed("hello world")
    assert isinstance(result, list)
    assert len(result) == 1024


@pytest.mark.asyncio
async def test_embed_raises_embedding_error_on_sdk_exception():
    from backend.services.embedding_service import embed, EmbeddingError
    mock_client = MagicMock()
    mock_client.embed.side_effect = RuntimeError("API error")
    with patch("backend.services.embedding_service.voyageai.Client", return_value=mock_client):
        with pytest.raises(EmbeddingError):
            await embed("some text")


@pytest.mark.asyncio
async def test_embed_document_input_type(mock_voyage_client):
    from backend.services.embedding_service import embed
    from unittest.mock import patch
    _, mock_client = mock_voyage_client
    with patch("backend.services.embedding_service.get_effective_config", return_value={
        "voyage_api_key": "test-key",
        "voyage_model_document": "voyage-3",
        "voyage_model_query": "voyage-3",
    }):
        await embed("transcript", input_type="document")
    mock_client.embed.assert_called_once_with(["transcript"], model="voyage-3", input_type="document")


@pytest.mark.asyncio
async def test_embed_query_input_type(mock_voyage_client):
    from backend.services.embedding_service import embed
    from unittest.mock import patch
    _, mock_client = mock_voyage_client
    with patch("backend.services.embedding_service.get_effective_config", return_value={
        "voyage_api_key": "test-key",
        "voyage_model_document": "voyage-3",
        "voyage_model_query": "voyage-3",
    }):
        await embed("query", input_type="query")
    mock_client.embed.assert_called_once_with(["query"], model="voyage-3", input_type="query")
