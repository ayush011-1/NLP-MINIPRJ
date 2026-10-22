"""ManSanket (मनसंकेत) - Emotion Analysis Workspace.

A simple, user-friendly emotion analyzer designed for everyday users:
- Clear text input with helpful placeholders
- Quick sample sentence selector (populates without auto-triggering)
- Focus on the likely emotion with plain-language explanations
- Compact 'Why this result?' feature words
- Optional expandable breakdown for estimated probabilities
- Honest, clear language notices and optional translation assistant
"""
import streamlit as st
import html
from src.config import PALETTE, EMOTION_COLORS
from src.i18n import t
from src.ui import render_sidebar, render_header
from src.predict import predict_emotion
from src.visualizations import plot_probability_bars


def render_analyzer_page():
    """Render the user-friendly emotion analyzer page."""
    render_sidebar()
    locale = st.session_state.get("locale", "en")

    render_header(
        title=t("analyzer.title", locale),
        subtitle=t("analyzer.subtitle", locale)
    )

    # Preset sample sentences covering everyday situations
    SAMPLES = {
        "-- " + t("common.select_sample", locale) + " --": "",
        t("analyzer.samples.joy", locale): "I feel so genuinely happy, delighted, and energized by this wonderful news!",
        t("analyzer.samples.sadness", locale): "I am feeling utterly broken, exhausted, and deeply disheartened today.",
        t("analyzer.samples.love", locale): "I cherish every moment spent with you, thank you for your warmth and care.",
        t("analyzer.samples.anger", locale): "I am furious and absolutely disgusted by this unfair and corrupt decision.",
        t("analyzer.samples.fear", locale): "I feel so anxious and terrified about what could go wrong tomorrow.",
        t("analyzer.samples.surprise", locale): "I was completely stunned and astonished when I opened the unexpected letter.",
        t("analyzer.samples.mr_joy", locale): "आज निकाल लागला आणि मला मनापासून खूप आनंद झाला आहे!",
        t("analyzer.samples.hi_anger", locale): "मुझे इस घटिया और झूठे बर्ताव पर बहुत ज्यादा गुस्सा आ रहा है!",
        t("analyzer.samples.hinglish_fear", locale): "Bohot zyada darr lag raha hai exam ke results ko lekar yaar."
    }

    # Session state initialization for input text
    if "user_text_content" not in st.session_state:
        st.session_state["user_text_content"] = ""
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "last_analyzed_query" not in st.session_state:
        st.session_state["last_analyzed_query"] = ""

    # Sample selector (populates input without auto-analyzing)
    selected_sample = st.selectbox(
        t("analyzer.sample_prompt", locale),
        options=list(SAMPLES.keys()),
        index=0,
        key="sample_selector_widget"
    )

    # When user picks a non-empty sample, populate the input state
    if selected_sample and SAMPLES[selected_sample]:
        sample_text = SAMPLES[selected_sample]
        if st.session_state["user_text_content"] != sample_text:
            st.session_state["user_text_content"] = sample_text
            st.session_state["analysis_result"] = None

    # Text input area
    input_text = st.text_area(
        label=t("common.input_label", locale),
        value=st.session_state["user_text_content"],
        height=130,
        placeholder=t("common.input_placeholder", locale),
        key="text_input_area"
    )
    # Sync typing into session state
    st.session_state["user_text_content"] = input_text

    # Advanced Settings (collapsed by default so everyday users aren't confused)
    with st.expander(t("analyzer.advanced_options", locale), expanded=False):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            manual_lang = st.selectbox(
                t("analyzer.lang_override_label", locale),
                options=["auto", "en", "hi", "mr", "hinglish", "mr-en-codemixed"],
                format_func=lambda x: {
                    "auto": t("analyzer.auto_detect_label", locale),
                    "en": "English (en)",
                    "hi": "Hindi (hi - Devanagari)",
                    "mr": "Marathi (mr - Devanagari)",
                    "hinglish": "Hinglish (Romanized Hindi)",
                    "mr-en-codemixed": "Marathi-English (Romanized Marathi)"
                }.get(x, x),
                index=0,
                key="manual_language_override"
            )
            classifier_choice = st.selectbox(
                t("analyzer.model_select_label", locale),
                options=["Linear SVC", "Logistic Regression", "Multinomial Naive Bayes"],
                index=0,
                key="classifier_choice_widget"
            )
        with col_adv2:
            st.markdown(f"**{t('analyzer.translation_title', locale)}**")
            st.caption(t("analyzer.translation_desc", locale))
            allow_translation = st.checkbox(
                t("analyzer.translation_consent", locale),
                value=False,
                key="translation_consent_checkbox"
            )

    # Action buttons
    col_act1, col_act2, _ = st.columns([1.3, 1.0, 2.5])
    with col_act1:
        analyze_clicked = st.button(
            t("common.analyze_btn", locale),
            key="analyze_emotion_button",
            use_container_width=True
        )
    with col_act2:
        if st.button(t("common.clear_btn", locale), key="clear_emotion_button", use_container_width=True):
            st.session_state["user_text_content"] = ""
            st.session_state["analysis_result"] = None
            st.session_state["last_analyzed_query"] = ""
            st.rerun()

    # Process Analysis
    if analyze_clicked:
        current_text = st.session_state["user_text_content"].strip()
        if not current_text:
            st.warning(t("common.no_input_warn", locale))
            st.session_state["analysis_result"] = None
        else:
            with st.spinner(t("common.loading", locale)):
                result = predict_emotion(
                    text=current_text,
                    model_name=classifier_choice,
                    manual_lang=manual_lang,
                    allow_translation=allow_translation
                )
                st.session_state["analysis_result"] = result
                st.session_state["last_analyzed_query"] = current_text

    # Display Results if Available
    res = st.session_state.get("analysis_result")
    if res and res.get("is_valid"):
        pred_emotion = res["predicted_label"]
        confidence = res["confidence"]
        lang_info = res["language_info"]
        detected_code = res.get("effective_lang", lang_info["language_code"])
        color = EMOTION_COLORS.get(pred_emotion, PALETTE["primary_accent"])

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # 1. Honest Language Route Notice
        if res.get("is_translation_mediated"):
            trans_safe = html.escape(res.get("translated_text", ""))
            st.info(t("analyzer.translation_active_notice", locale, translated=trans_safe))
        elif res.get("translation_note"):
            st.warning(t("analyzer.translation_failed_notice", locale))
        elif detected_code in ["hi", "mr", "hinglish", "mr-en-codemixed", "code-mixed"]:
            lang_display_name = lang_info.get("language_name", detected_code)
            st.markdown(
                f"""
                <div class="editorial-card-alt" style="border-left: 4px solid {PALETTE['warning']}; padding: 0.85rem 1.15rem; margin-bottom: 1rem; font-size: 0.88rem; color: {PALETTE['text_secondary']};">
                    {t('analyzer.non_english_direct_notice', locale, lang=lang_display_name)}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # English direct
            st.caption(t("analyzer.english_direct_notice", locale))

        # 2. Main Emotion Result Card
        localized_emotion_name = t(f"emotions.{pred_emotion}", locale)
        emotion_description = t(f"emotion_descriptions.{pred_emotion}", locale)
        confidence_percent = int(round(confidence * 100))

        st.markdown(
            f"""
            <div class="emotion-display-card" style="border-left: 5px solid {color}; border-top: 1px solid {PALETTE['border']}; border-right: 1px solid {PALETTE['border']}; border-bottom: 1px solid {PALETTE['border']};">
                <div style="font-size: 0.82rem; color: {PALETTE['text_secondary']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;">
                    {t('analyzer.likely_emotion_badge', locale)}
                </div>
                <div style="display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 0.6rem;">
                    <span style="font-size: 2.25rem; font-weight: 700; color: {color}; letter-spacing: 0.01em;">
                        {localized_emotion_name}
                    </span>
                    <span style="font-size: 1rem; font-weight: 500; color: {PALETTE['text_secondary']};">
                        ({confidence_percent}% confidence estimate)
                    </span>
                </div>
                <div style="font-size: 0.95rem; line-height: 1.55; color: {PALETTE['text_primary']};">
                    {emotion_description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 3. Compact "Why this result?"
        top_words = [f["feature"] for f in res.get("top_features", []) if f.get("contribution", 0) > 0][:5]
        token_attribs = res.get("token_attributions", [])

        st.markdown(
            f"""
            <div class="editorial-card" style="padding: 1.25rem 1.5rem; margin-bottom: 1rem;">
                <div style="font-weight: 600; font-size: 1rem; color: {PALETTE['primary_accent']}; margin-bottom: 0.5rem;">
                    {t('analyzer.why_title', locale)}
                </div>
            """,
            unsafe_allow_html=True
        )

        if top_words:
            # Render influential words in soft, readable badges
            badges_html = " ".join([
                f'<span style="display: inline-block; background-color: {PALETTE["surface_alt"]}; border: 1px solid {PALETTE["border"]}; color: {PALETTE["primary_accent"]}; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 4px; margin: 0.15rem 0.2rem; font-size: 0.88rem;">{html.escape(w)}</span>'
                for w in top_words
            ])
            st.markdown(
                f"""
                <div style="font-size: 0.92rem; color: {PALETTE['text_secondary']}; margin-bottom: 0.6rem;">
                    {t('analyzer.influential_words_intro', locale)}
                </div>
                <div style="margin-bottom: 0.75rem;">
                    {badges_html}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="font-size: 0.92rem; color: {PALETTE['text_secondary']};">
                    {t('analyzer.no_prominent_words', locale)}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # 4. Optional Expandable Breakdown (Clean & Compact)
        with st.expander(t("analyzer.breakdown_expander", locale), expanded=False):
            st.caption(t("analyzer.breakdown_note", locale))
            prob_dict = res.get("probabilities", {})
            if prob_dict:
                # Render clean, friendly progress rows
                for emo, prob in prob_dict.items():
                    emo_name = t(f"emotions.{emo}", locale)
                    bar_color = EMOTION_COLORS.get(emo, PALETTE["primary_accent"])
                    pct = int(round(prob * 100))
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 0.65rem;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.2rem;">
                                <span style="font-weight: 500; color: {PALETTE['text_primary']};">{emo_name}</span>
                                <span style="color: {PALETTE['text_secondary']}; font-weight: 600;">{pct}%</span>
                            </div>
                            <div style="background-color: {PALETTE['surface_alt']}; border: 1px solid {PALETTE['border']}; border-radius: 4px; height: 10px; width: 100%; overflow: hidden;">
                                <div style="background-color: {bar_color}; width: {max(2, pct)}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# Standalone runner for direct page navigation
if __name__ == "__main__" or True:
    render_analyzer_page()
