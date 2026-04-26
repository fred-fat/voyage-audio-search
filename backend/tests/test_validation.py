"""
Property-based tests for validation utilities in backend/models.py.
"""
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from backend.models import (
    ALLOWED_EXTENSIONS,
    validate_audio_extension,
    validate_query_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_EXTENSIONS = list(ALLOWED_EXTENSIONS)  # [".mp3", ".wav", ".m4a", ".flac"]


# ---------------------------------------------------------------------------
# Property 2: Invalid format validation
# Validates: Requirements 1.7
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.text().filter(
    lambda s: s.lower() not in ALLOWED_EXTENSIONS
))
def test_property2_invalid_extension_returns_false(ext):
    """
    **Validates: Requirements 1.7**

    Property 2: For any string that is not a valid audio extension,
    validate_audio_extension("file" + ext) should return False.
    """
    filename = "testfile" + ext
    assert validate_audio_extension(filename) is False


@pytest.mark.parametrize("ext", VALID_EXTENSIONS)
def test_property2_valid_extension_returns_true(ext):
    """
    Property 2 (positive case): Each allowed extension should return True,
    both in lowercase and uppercase.
    """
    assert validate_audio_extension("audio" + ext) is True
    assert validate_audio_extension("audio" + ext.upper()) is True


# ---------------------------------------------------------------------------
# Property 6: Whitespace-only queries are rejected
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
@given(st.text(alphabet=st.characters(categories=["Zs", "Cc"])).filter(
    lambda s: len(s) > 0 and s.strip() == ""
))
def test_property6_whitespace_query_returns_false(query):
    """
    **Validates: Requirements 3.4**

    Property 6: For any string composed entirely of whitespace characters,
    validate_query_text should return False.
    """
    assert validate_query_text(query) is False


@settings(max_examples=100)
@given(st.text(alphabet=" \t\n\r").filter(lambda s: len(s) > 0))
def test_property6_explicit_whitespace_chars_rejected(query):
    """
    **Validates: Requirements 3.4**

    Property 6 (explicit whitespace alphabet): Using space/tab/newline/CR,
    validate_query_text should return False for any non-empty whitespace-only string.
    """
    assert validate_query_text(query) is False


@settings(max_examples=100)
@given(st.text(min_size=1).filter(lambda s: s.strip() != ""))
def test_property6_non_empty_query_returns_true(query):
    """
    Property 6 (positive case): Any string with at least one non-whitespace
    character should be accepted by validate_query_text.
    """
    assert validate_query_text(query) is True


def test_property6_empty_string_rejected():
    """Edge case: empty string should be rejected."""
    assert validate_query_text("") is False
