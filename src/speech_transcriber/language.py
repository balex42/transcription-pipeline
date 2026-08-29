"""Small backend-neutral helpers for configured ASR languages."""

from __future__ import annotations


def normalize_language(language: str | None) -> str | None:
    """Reduce a locale such as ``de-DE`` to its base language code."""
    if language is None:
        return None
    return language.split("-", 1)[0].lower()
