from pydantic import BaseModel

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}


def validate_audio_extension(filename: str) -> bool:
    """Check if the file extension is in ALLOWED_EXTENSIONS (case-insensitive)."""
    if not filename:
        return False
    dot_index = filename.rfind(".")
    if dot_index == -1:
        return False
    ext = filename[dot_index:].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_query_text(query: str) -> bool:
    """Reject empty strings and whitespace-only strings."""
    if not query or not query.strip():
        return False
    return True


class SearchTextRequest(BaseModel):
    query: str


class AudioRecordResponse(BaseModel):
    id: str
    filename: str
    transcript: str
    score: float


class SearchResponse(BaseModel):
    query_transcript: str | None = None
    results: list[AudioRecordResponse]


class IngestResponse(BaseModel):
    id: str
    filename: str
    transcript: str
