# Voyage Audio Search

A semantic audio search demo built with Voyage AI embeddings + MongoDB Atlas Vector Search. Upload audio files, transcribe them locally with mlx-whisper (optimized for Apple Silicon), and search by audio or text query.

## Requirements

- Python 3.12+
- Apple Silicon Mac (M-series) — required for mlx-whisper local transcription
- [MongoDB Atlas](https://www.mongodb.com/atlas) account with a cluster
- [Voyage AI](https://www.voyageai.com) API Key
- [ffmpeg](https://ffmpeg.org) — required by mlx-whisper for audio decoding

```bash
brew install ffmpeg
```

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

### Option 1: UI Settings Panel (recommended)

Start the frontend, then open the **⚙️ Settings** sidebar to enter your `Voyage API Key` and `MongoDB URI`. Settings are saved to `config.local.json` locally and never uploaded to GitHub.

### Option 2: Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in:
# VOYAGE_API_KEY=your_voyage_api_key_here
# MONGODB_URI=mongodb+srv://...
```

## Create MongoDB Atlas Vector Index

Before first use, create the vector search index on your Atlas cluster:

```bash
venv/bin/python backend/scripts/create_index.py
```

> Index build takes a few minutes. Wait until it shows **Active** in the Atlas UI before searching.

## Start the Services

```bash
# Terminal 1 — Backend (FastAPI)
venv/bin/uvicorn backend.main:app --reload

# Terminal 2 — Frontend (Streamlit)
venv/bin/streamlit run frontend/app.py
```

## Run Tests

```bash
venv/bin/pytest backend/tests/ -v
```

## Project Structure

```
voyage-audio-search/
├── backend/
│   ├── services/
│   │   ├── stt_service.py        # mlx-whisper transcription
│   │   ├── embedding_service.py  # Voyage AI embeddings
│   │   └── vector_store.py       # MongoDB Atlas vector search
│   ├── scripts/
│   │   └── create_index.py       # One-time index creation
│   ├── tests/                    # Unit + property-based tests
│   ├── config_service.py         # Config management (local file + env vars)
│   ├── debug_log.py              # In-memory debug log buffer
│   ├── main.py                   # FastAPI app
│   └── models.py                 # Pydantic models + validation
├── frontend/
│   └── app.py                    # Streamlit UI
├── .env.example                  # Environment variable template
├── requirements.txt
└── README.md
```

## Security

`config.local.json` is listed in `.gitignore` and will never be committed. Your API keys and database credentials stay local.
