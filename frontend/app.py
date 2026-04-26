import json
import os
import re
import sys
from pathlib import Path

import requests
import streamlit as st

# Add project root to sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config_service import get_effective_config, save_config


def is_configured() -> bool:
    c = get_effective_config()
    return bool(c.get("voyage_api_key")) and bool(c.get("mongodb_uri"))


def mask_mongodb_uri(uri: str) -> str:
    """Mask the password in a MongoDB URI for display purposes."""
    if not uri:
        return ""
    # mongodb+srv://user:password@host/... → mongodb+srv://user:***@host/...
    return re.sub(r"(mongodb(?:\+srv)?://[^:]+:)([^@]+)(@)", r"\1***\3", uri)


def _render_results(results: list) -> None:
    if not results:
        st.info("No records found. Please ingest audio files first.")
        return
    for i, item in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i}. {item.get('filename', 'Unknown')}**")
                st.markdown(item.get("transcript", ""))
            with col2:
                score = item.get("score", 0)
                st.metric("Score", f"{score:.4f}")
            st.divider()


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(page_title="Voyage Audio Search", page_icon="🎵", layout="wide")
st.title("🎵 Voyage Audio Search")

# ──────────────────────────────────────────────
# Sidebar: Settings
# ──────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

# Voyage AI embedding models
VOYAGE_MODELS = [
    "voyage-4-large",   # Best general-purpose & multilingual, 256/512/1024/2048 dims
    "voyage-4",         # General-purpose & multilingual, 256/512/1024/2048 dims
    "voyage-4-lite",    # Low latency & cost, 256/512/1024/2048 dims
]

FLEXIBLE_DIM_MODELS = {"voyage-4-large", "voyage-4", "voyage-4-lite"}

cfg = get_effective_config()
saved_key = cfg.get("voyage_api_key") or ""
saved_uri = cfg.get("mongodb_uri") or ""

# ── Voyage AI ──
st.sidebar.subheader("🔑 Voyage AI")
key_placeholder = "••••••••" if saved_key else ""
voyage_key_input = st.sidebar.text_input(
    "API Key",
    value=key_placeholder,
    type="password",
    help="Your Voyage AI API Key",
)

# Ingest model is fixed — do not change
INGEST_MODEL = "voyage-4-large"
INGEST_DIMENSIONS = 1024
voyage_doc_model = INGEST_MODEL

st.sidebar.text_input(
    "Ingest Embedding Model (Document)",
    value=INGEST_MODEL,
    disabled=True,
    help="Fixed for this demo. All ingested audio uses voyage-4-large at 1024 dimensions.",
)
st.sidebar.text_input(
    "Ingest Embedding Dimensions",
    value=str(INGEST_DIMENSIONS),
    disabled=True,
    help="Fixed at 1024 to match the MongoDB Atlas vector index.",
)
st.sidebar.caption(
    "ℹ️ This demo uses **voyage-4-large** at **1024 dimensions** for ingestion. "
    "Changing these would require re-ingesting all audio and recreating the vector index."
)

# Query model is configurable
query_model_default = cfg.get("voyage_model_query", "voyage-4-large")
query_model_idx = VOYAGE_MODELS.index(query_model_default) if query_model_default in VOYAGE_MODELS else 0
voyage_query_model = st.sidebar.selectbox(
    "Query Embedding Model",
    options=VOYAGE_MODELS,
    index=query_model_idx,
    help="Model used to embed queries during search. Must be compatible with the ingest model.",
)

embedding_dim_input = INGEST_DIMENSIONS

# ── MongoDB Atlas ──
st.sidebar.subheader("🍃 MongoDB Atlas")

# Show masked URI as placeholder when already saved
uri_display = mask_mongodb_uri(saved_uri) if saved_uri else ""
mongodb_uri_input = st.sidebar.text_input(
    "Connection String (URI)",
    value=uri_display,
    help="mongodb+srv://user:password@cluster.mongodb.net/",
    placeholder="mongodb+srv://...",
)

mongodb_db_input = st.sidebar.text_input(
    "Database Name",
    value=cfg.get("mongodb_db", "voyage_audio_search"),
    disabled=True,
)
mongodb_col_input = st.sidebar.text_input(
    "Collection Name",
    value=cfg.get("mongodb_collection", "audio_records"),
    disabled=True,
)
search_top_k_input = st.sidebar.number_input(
    "Search Top-K Results",
    min_value=1,
    max_value=20,
    value=int(cfg.get("search_top_k", 5)),
    step=1,
)

# ── STT ──
st.sidebar.subheader("🎙️ STT (Whisper)")
whisper_model_input = st.sidebar.text_input(
    "Whisper Model",
    value=cfg.get("whisper_model", "mlx-community/whisper-large-v3-mlx"),
    help="e.g. mlx-community/whisper-large-v3-mlx or mlx-community/whisper-small-mlx",
)

# ── Backend ──
st.sidebar.subheader("🔧 Backend")
backend_url_input = st.sidebar.text_input(
    "Backend URL",
    value=cfg.get("backend_url", "http://localhost:8000"),
)

# ── Save / Validate buttons ──
col_save, col_validate = st.sidebar.columns(2)
with col_save:
    if st.button("💾 Save", key="save_btn"):
        actual_key = voyage_key_input if voyage_key_input != "••••••••" else saved_key
        # If URI field still shows masked value, keep the original saved URI
        actual_uri = mongodb_uri_input.strip()
        if actual_uri == uri_display and saved_uri:
            actual_uri = saved_uri
        new_config = {
            "voyage_api_key": actual_key,
            "voyage_model_document": voyage_doc_model,
            "voyage_model_query": voyage_query_model,
            "embedding_dimensions": int(embedding_dim_input),
            "mongodb_uri": actual_uri,
            "mongodb_db": mongodb_db_input.strip(),
            "mongodb_collection": mongodb_col_input.strip(),
            "search_top_k": int(search_top_k_input),
            "whisper_model": whisper_model_input.strip(),
            "backend_url": backend_url_input.strip(),
        }
        save_config(new_config)
        st.sidebar.success("✅ Settings saved")

with col_validate:
    if st.button("🔍 Validate", key="validate_btn"):
        actual_key = voyage_key_input if voyage_key_input != "••••••••" else saved_key
        if not actual_key:
            st.sidebar.warning("Please enter your Voyage API Key first")
        else:
            with st.sidebar:
                with st.spinner("Validating..."):
                    try:
                        resp = requests.post(
                            f"{get_effective_config().get('backend_url', 'http://localhost:8000')}/validate-config",
                            json={
                                "voyage_api_key": actual_key,
                                "voyage_model_document": voyage_doc_model,
                                "voyage_model_query": voyage_query_model,
                            },
                            timeout=30,
                        )
                        data = resp.json()
                        doc_r = data["results"]["document_model"]
                        qry_r = data["results"]["query_model"]
                        if data["all_ok"]:
                            st.success("✅ API Key valid, models available")
                        else:
                            if not doc_r["ok"]:
                                st.error(f"❌ Ingest model {doc_r['model']}: {doc_r.get('error', 'unavailable')}")
                            if not qry_r["ok"]:
                                st.error(f"❌ Query model {qry_r['model']}: {qry_r.get('error', 'unavailable')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Please start the backend first.")

# Reload backend URL after potential save
BACKEND_URL = get_effective_config().get("backend_url", "http://localhost:8000")

if not cfg.get("voyage_api_key") and not cfg.get("mongodb_uri"):
    st.sidebar.warning("⚠️ Please configure your API Key and MongoDB URI")

# ──────────────────────────────────────────────
# Main area: Tabs
# ──────────────────────────────────────────────
tab_ingest, tab_search = st.tabs(["📥 Ingest Audio", "🔍 Search"])

# ──────────────────────────────────────────────
# Tab 1: Ingest Audio
# ──────────────────────────────────────────────
with tab_ingest:
    st.header("📥 Ingest Audio")
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "wav", "m4a", "flac"],
        key="ingest_uploader",
    )
    if st.button("Start Ingestion", key="ingest_btn"):
        if not is_configured():
            st.warning("Please configure your API Key and MongoDB URI in the sidebar first.")
        elif uploaded_file is None:
            st.warning("Please select an audio file.")
        else:
            with st.spinner("Processing..."):
                try:
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    resp = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ Ingestion successful!\n\n**Transcript:** {data.get('transcript')}")
                    else:
                        try:
                            detail = resp.json().get("detail", resp.text)
                        except Exception:
                            detail = resp.text
                        st.error(f"Error: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Please start the backend first.")
                except Exception as e:
                    st.error(f"Request failed: {e}")

# ──────────────────────────────────────────────
# Tab 2: Search
# ──────────────────────────────────────────────
with tab_search:
    st.header("🔍 Search")
    search_tab_audio, search_tab_text = st.tabs(["🎵 Audio Search", "📝 Text Search"])

    with search_tab_audio:
        search_audio_file = st.file_uploader(
            "Upload a query audio file",
            type=["mp3", "wav", "m4a", "flac"],
            key="search_audio_uploader",
        )
        if st.button("Search", key="search_audio_btn"):
            if not is_configured():
                st.warning("Please configure your API Key and MongoDB URI in the sidebar first.")
            elif search_audio_file is None:
                st.warning("Please select an audio file.")
            else:
                with st.spinner("Searching..."):
                    try:
                        files = {
                            "file": (
                                search_audio_file.name,
                                search_audio_file.getvalue(),
                                search_audio_file.type,
                            )
                        }
                        resp = requests.post(f"{BACKEND_URL}/search/audio", files=files, timeout=120)
                        if resp.status_code == 200:
                            data = resp.json()
                            results = data.get("results", [])
                            query_transcript = data.get("query_transcript", "")
                            if query_transcript:
                                st.info(f"Query transcript: {query_transcript}")
                            _render_results(results)
                        else:
                            try:
                                detail = resp.json().get("detail", resp.text)
                            except Exception:
                                detail = resp.text
                            st.error(f"Error: {detail}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Please start the backend first.")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

    with search_tab_text:
        query_text = st.text_input("Enter search text", key="search_text_input")
        if st.button("Search", key="search_text_btn"):
            if not is_configured():
                st.warning("Please configure your API Key and MongoDB URI in the sidebar first.")
            elif not query_text.strip():
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Searching..."):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/search/text",
                            json={"query": query_text},
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            results = data.get("results", [])
                            _render_results(results)
                        else:
                            try:
                                detail = resp.json().get("detail", resp.text)
                            except Exception:
                                detail = resp.text
                            st.error(f"Error: {detail}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Please start the backend first.")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

# ──────────────────────────────────────────────
# Debug Log panel
# ──────────────────────────────────────────────
st.divider()
with st.expander("🐛 Debug Log", expanded=False):
    col_refresh, col_clear, col_limit = st.columns([1, 1, 2])
    with col_refresh:
        st.button("🔄 Refresh", key="debug_refresh")
    with col_clear:
        if st.button("🗑️ Clear", key="debug_clear"):
            try:
                requests.post(f"{BACKEND_URL}/debug/clear", timeout=5)
                st.success("Cleared")
            except Exception:
                st.warning("Failed to clear logs")
    with col_limit:
        log_limit = st.slider("Max entries", min_value=10, max_value=200, value=50, step=10, key="log_limit")

    try:
        resp = requests.get(f"{BACKEND_URL}/debug/logs?limit={log_limit}", timeout=5)
        logs = resp.json().get("logs", [])
        if not logs:
            st.info("No logs yet. Run an ingest or search operation to see detailed execution info.")
        else:
            level_icon = {"INFO": "🟢", "DEBUG": "🔵", "ERROR": "🔴"}
            cat_icon = {"API": "🌐", "STT": "🎙️", "EMBED": "🧮", "MONGO": "🍃"}
            lines = []
            for entry in logs:
                icon = level_icon.get(entry["level"], "⚪")
                cat = cat_icon.get(entry["category"], "📋")
                line = f"{icon} [{entry['ts']}] {cat} {entry['category']} | {entry['message']}"
                if entry.get("detail"):
                    detail_str = json.dumps(entry["detail"], ensure_ascii=False)
                    line += f"\n    {detail_str}"
                lines.append(line)
            st.code("\n".join(lines), language="text")
    except requests.exceptions.ConnectionError:
        st.warning("Cannot connect to backend. Please start the backend first.")
    except Exception as e:
        st.warning(f"Failed to fetch logs: {e}")
