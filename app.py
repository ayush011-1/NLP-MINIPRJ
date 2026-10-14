"""ManSanket (मनसंकेत) - Main Entrypoint & Application Orchestration.

Multilingual NLP Emotion Detection System:
- Clean, user-friendly navigation for everyday users (Home & Emotion Analyzer)
- Collapsed academic and technical details for students, researchers, and examiners
- Localized across English, Hindi, and Marathi
"""
import streamlit as st
from src.config import DEFAULT_LOCALE
from src.i18n import t
from src.ui import setup_page
from src.views.home import render_home_page

# Initialize base page settings
setup_page("Home")

locale = st.session_state.get("locale", DEFAULT_LOCALE)

# Define clean two-tier navigation structure
pages = {
    t("app.title", locale): [
        st.Page(
            render_home_page,
            title=t("nav.home", locale),
            icon="🏠",
            default=True
        ),
        st.Page(
            "pages/1_Emotion_Analysis.py",
            title=t("nav.analyzer", locale),
            icon="✨"
        )
    ],
    t("nav.academic_section", locale): [
        st.Page(
            "pages/2_Model_Evaluation.py",
            title=t("nav.evaluation", locale),
            icon="📊"
        ),
        st.Page(
            "pages/3_Dataset_Explorer.py",
            title=t("nav.explorer", locale),
            icon="📁"
        ),
        st.Page(
            "pages/4_Model_Explainability.py",
            title=t("nav.explainability", locale),
            icon="🔍"
        ),
        st.Page(
            "pages/5_Methodology.py",
            title=t("nav.methodology", locale),
            icon="📚"
        )
    ]
}

# Run navigation with academic details collapsed by default
nav = st.navigation(pages, expanded=False)
nav.run()
