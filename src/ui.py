"""Shared UI components, styling, and layout helpers for ManSanket (मनसंकेत).

Provides a welcoming, polished, human-designed experience:
- Clean typography and comfortable spacing
- Curated emotion styling
- Centralized language selection
- Accessible contrast and responsive layout
"""
import streamlit as st
from typing import Optional, Dict, Any
from src.config import PALETTE, EMOTION_COLORS, SUPPORTED_LOCALES, DEFAULT_LOCALE
from src.i18n import t

# Clean, polished editorial CSS
EDITORIAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@300;400;500;600;700&family=Noto+Serif+Devanagari:wght@400;600;700&display=swap');

/* Main app background and typography */
.stApp {{
    background-color: {PALETTE["bg"]};
    color: {PALETTE["text_primary"]};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

/* Main container max width and padding */
.main .block-container {{
    max-width: 960px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}}

/* Clean, friendly header bar */
.editorial-header {{
    border-bottom: 1px solid {PALETTE["border"]};
    padding-bottom: 1.25rem;
    margin-bottom: 1.75rem;
}}

.editorial-wordmark {{
    font-family: 'Noto Serif Devanagari', 'Cinzel', Georgia, serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: {PALETTE["primary_accent"]};
    letter-spacing: 0.03em;
    display: inline-block;
}}

.editorial-english-title {{
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: {PALETTE["supporting_accent"]};
    margin-left: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.editorial-tagline {{
    font-size: 1rem;
    color: {PALETTE["text_secondary"]};
    margin-top: 0.25rem;
    font-weight: 400;
}}

/* Friendly rounded card container */
.editorial-card {{
    background-color: {PALETTE["surface"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}}

/* Alternate subtle card */
.editorial-card-alt {{
    background-color: {PALETTE["surface_alt"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}}

/* Emotion highlight card */
.emotion-display-card {{
    background-color: {PALETTE["surface"]};
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
}}

/* Primary Buttons */
.stButton > button {{
    background-color: {PALETTE["primary_accent"]} !important;
    color: #FFFFFF !important;
    border: 1px solid {PALETTE["primary_accent"]} !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    font-size: 0.95rem !important;
}}

.stButton > button:hover {{
    background-color: #243243 !important;
    border-color: #243243 !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
}}

/* Secondary / Clear Buttons */
button[kind="secondary"] {{
    background-color: #FFFFFF !important;
    color: {PALETTE["text_primary"]} !important;
    border: 1px solid {PALETTE["border"]} !important;
}}

button[kind="secondary"]:hover {{
    background-color: {PALETTE["surface_alt"]} !important;
    border-color: {PALETTE["text_secondary"]} !important;
}}

/* Form inputs */
.stTextArea textarea, .stTextInput input {{
    background-color: {PALETTE["surface"]} !important;
    border: 1px solid {PALETTE["border"]} !important;
    color: {PALETTE["text_primary"]} !important;
    border-radius: 6px !important;
    font-size: 0.98rem !important;
}}

.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: {PALETTE["primary_accent"]} !important;
    box-shadow: 0 0 0 1px {PALETTE["primary_accent"]} !important;
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background-color: #F2EFE8;
    border-right: 1px solid {PALETTE["border"]};
}}

/* Clean tables */
.editorial-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}}

.editorial-table th {{
    background-color: {PALETTE["surface_alt"]};
    color: {PALETTE["primary_accent"]};
    font-weight: 600;
    text-align: left;
    padding: 0.65rem 0.85rem;
    border-bottom: 2px solid {PALETTE["border"]};
}}

.editorial-table td {{
    padding: 0.6rem 0.85rem;
    border-bottom: 1px solid {PALETTE["border"]};
    color: {PALETTE["text_primary"]};
}}

.editorial-table tr:hover {{
    background-color: #FAF8F5;
}}
</style>
"""


def setup_page(title: str):
    """Initialize page metadata, session state, and custom styling."""
    try:
        st.set_page_config(
            page_title=f"{title} | ManSanket (मनसंकेत)",
            page_icon="🌸",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        # Ignore if st.set_page_config was already called by entrypoint
        pass
    st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)
    
    # Initialize locale in session state
    if "locale" not in st.session_state:
        st.session_state["locale"] = DEFAULT_LOCALE


def render_sidebar():
    """Render language selector and friendly identity in the sidebar."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding-bottom: 0.75rem; border-bottom: 1px solid {PALETTE['border']}; margin-bottom: 1rem;">
                <div style="font-family: 'Noto Serif Devanagari', Georgia, serif; font-size: 1.45rem; font-weight: 700; color: {PALETTE['primary_accent']};">मनसंकेत</div>
                <div style="font-size: 0.8rem; color: {PALETTE['supporting_accent']}; font-weight: 600; text-transform: uppercase;">ManSanket</div>
                <div style="font-size: 0.8rem; color: {PALETTE['text_secondary']}; margin-top: 0.15rem;">Text Emotion Detection</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        current_locale = st.session_state.get("locale", DEFAULT_LOCALE)
        locale_options = {
            "en": "English",
            "hi": "हिंदी (Hindi)",
            "mr": "मराठी (Marathi)"
        }
        
        selected_code = st.selectbox(
            f"🌐 {t('common.language', current_locale)}",
            options=list(locale_options.keys()),
            format_func=lambda x: locale_options[x],
            index=list(locale_options.keys()).index(current_locale),
            key="sidebar_language_selector"
        )
        if selected_code != current_locale:
            st.session_state["locale"] = selected_code
            st.rerun()


def render_header(title: Optional[str] = None, subtitle: Optional[str] = None):
    """Render the welcoming application header."""
    locale = st.session_state.get("locale", DEFAULT_LOCALE)
    title_html = f'<h1 style="font-size: 1.7rem; font-weight: 700; color: {PALETTE["primary_accent"]}; margin-top: 0.6rem; margin-bottom: 0.2rem;">{title}</h1>' if title else ''
    sub_html = f'<div style="font-size: 0.95rem; color: {PALETTE["text_secondary"]};">{subtitle}</div>' if subtitle else ''
    
    st.markdown(
        f"""
        <div class="editorial-header">
            <div>
                <span class="editorial-wordmark">मनसंकेत</span>
                <span class="editorial-english-title">ManSanket</span>
            </div>
            <div class="editorial-tagline">{t('app.tagline', locale)}</div>
            {title_html}
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )
