"""Internationalization (i18n) module for ManSanket (मनसंकेत).

Provides centralized translation loading, localized text retrieval with fallback,
and language state management for English, Hindi, and Marathi.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from src.config import LOCALES_DIR, SUPPORTED_LOCALES, DEFAULT_LOCALE

_TRANSLATIONS: Dict[str, Dict[str, Any]] = {}


def load_translations() -> Dict[str, Dict[str, Any]]:
    """Load translation dictionaries for all supported locales from JSON files."""
    global _TRANSLATIONS
    if _TRANSLATIONS:
        return _TRANSLATIONS

    for locale in SUPPORTED_LOCALES:
        locale_file = LOCALES_DIR / f"{locale}.json"
        if locale_file.exists():
            try:
                with open(locale_file, "r", encoding="utf-8") as f:
                    _TRANSLATIONS[locale] = json.load(f)
            except Exception as e:
                print(f"Error loading locale {locale}: {e}")
                _TRANSLATIONS[locale] = {}
        else:
            _TRANSLATIONS[locale] = {}

    return _TRANSLATIONS


def t(key: str, locale: str = DEFAULT_LOCALE, **kwargs: Any) -> str:
    """Retrieve translated text for a given dotted or direct key with fallback to English.
    
    Args:
        key: Dotted key string (e.g., 'common.analyze_button') or direct key.
        locale: Target locale code ('en', 'hi', 'mr').
        **kwargs: Optional formatting keyword arguments.
        
    Returns:
        Formatted translated string.
    """
    translations = load_translations()
    
    # Ensure valid locale
    target_locale = locale if locale in SUPPORTED_LOCALES else DEFAULT_LOCALE
    
    # Helper to resolve dotted paths in nested dicts
    def get_nested(d: Dict[str, Any], path: str) -> Optional[str]:
        keys = path.split(".")
        val: Any = d
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return None
        return val if isinstance(val, str) else None

    # Try target locale
    text = get_nested(translations.get(target_locale, {}), key)
    
    # Fallback to English if not found
    if text is None and target_locale != DEFAULT_LOCALE:
        text = get_nested(translations.get(DEFAULT_LOCALE, {}), key)
        
    if text is None:
        # Return last segment or key if missing
        text = key.split(".")[-1]

    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text

    return text
