"""User-friendly Home Page for ManSanket (मनसंकेत).

Welcoming, simple introduction designed for everyday users:
- Clear description of what the tool does
- Prominent 'Analyze Text' call-to-action
- Overview of the 6 supported emotions with friendly descriptions
- Gentle note on how automated estimation works
"""
import streamlit as st
from src.config import PALETTE, EMOTION_COLORS
from src.i18n import t
from src.ui import render_sidebar, render_header


def render_home_page():
    """Render the user-focused Home page."""
    render_sidebar()
    locale = st.session_state.get("locale", "en")

    render_header(
        title=t("home.welcome_title", locale),
        subtitle=t("app.subtitle", locale)
    )

    # Friendly hero card
    st.markdown(
        f"""
        <div class="editorial-card" style="padding: 1.75rem 2rem; margin-bottom: 1.5rem;">
            <div style="font-size: 1.15rem; line-height: 1.65; color: {PALETTE['text_primary']}; margin-bottom: 1.25rem;">
                {t('home.welcome_lead', locale)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Prominent Call-To-Action Button
    col_cta1, col_cta2, _ = st.columns([1.5, 1.5, 2])
    with col_cta1:
        if st.button(t("home.cta_btn", locale), key="home_cta_btn", use_container_width=True):
            # Switch to Emotion Analyzer page
            try:
                st.switch_page("pages/1_Emotion_Analysis.py")
            except Exception:
                st.session_state["nav_target"] = "analyzer"
                st.rerun()

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # The 6 Emotions We Detect Section
    st.markdown(f"### {t('home.emotions_title', locale)}")
    st.markdown(
        f"<div style='font-size: 0.95rem; color: {PALETTE['text_secondary']}; margin-bottom: 1.25rem;'>{t('home.emotions_subtitle', locale)}</div>",
        unsafe_allow_html=True
    )

    emotion_items = [
        ("joy", t("emotions.joy", locale), t("emotion_descriptions.joy", locale)),
        ("sadness", t("emotions.sadness", locale), t("emotion_descriptions.sadness", locale)),
        ("love", t("emotions.love", locale), t("emotion_descriptions.love", locale)),
        ("anger", t("emotions.anger", locale), t("emotion_descriptions.anger", locale)),
        ("fear", t("emotions.fear", locale), t("emotion_descriptions.fear", locale)),
        ("surprise", t("emotions.surprise", locale), t("emotion_descriptions.surprise", locale)),
    ]

    # Render a clean, balanced 2-column card grid
    col1, col2 = st.columns(2)
    for idx, (emo_key, name, desc) in enumerate(emotion_items):
        target_col = col1 if idx % 2 == 0 else col2
        accent_color = EMOTION_COLORS.get(emo_key, PALETTE["primary_accent"])
        with target_col:
            st.markdown(
                f"""
                <div class="editorial-card" style="border-left: 4px solid {accent_color}; margin-bottom: 1rem; padding: 1.15rem 1.35rem;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: {accent_color}; margin-bottom: 0.35rem;">
                        {name}
                    </div>
                    <div style="font-size: 0.92rem; line-height: 1.55; color: {PALETTE['text_secondary']};">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Gentle Disclaimer
    st.markdown(
        f"""
        <div class="editorial-card-alt" style="border-left: 4px solid {PALETTE['supporting_accent']}; margin-top: 1rem;">
            <div style="font-weight: 600; color: {PALETTE['primary_accent']}; font-size: 0.95rem; margin-bottom: 0.35rem;">
                {t('home.disclaimer_title', locale)}
            </div>
            <div style="font-size: 0.9rem; line-height: 1.55; color: {PALETTE['text_secondary']};">
                {t('home.disclaimer_body', locale)}
            </div>
            <div style="margin-top: 0.75rem; padding-top: 0.6rem; border-top: 1px solid {PALETTE['border']}; font-size: 0.85rem; color: {PALETTE['text_muted']};">
                {t('home.academic_prompt', locale)} <b>{t('home.academic_link', locale)}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
