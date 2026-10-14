"""ManSanket (मनसंकेत) - Model Explainability & Feature Attribution.

Page 5: In-depth interpretability of linear TF-IDF classifiers,
mathematical feature attributions, and global emotion lexicons.
"""
import streamlit as st
import pandas as pd
from src.config import PALETTE, EMOTION_LABELS, EMOTION_COLORS
from src.i18n import t
from src.ui import setup_page, render_sidebar, render_header
from src.explainability import get_global_top_features, explain_prediction
from src.visualizations import plot_token_contributions

setup_page("Explainability")
render_sidebar()

locale = st.session_state.get("locale", "en")

render_header(
    title=t("explainability.title", locale),
    subtitle=t("explainability.subtitle", locale)
)

# 1. Mathematical Formulation & Concept
st.markdown(
    f"""
    <div class="editorial-card">
        <h3 style="font-size: 1.15rem; color: {PALETTE['primary_accent']}; margin-top: 0; margin-bottom: 0.5rem;">
            {t('explainability.how_it_works', locale)}
        </h3>
        <div style="font-size: 0.9rem; line-height: 1.65; color: {PALETTE['text_primary']};">
            In linear NLP classifiers (such as <b>Linear Support Vector Machines</b> and <b>Multinomial Logistic Regression</b>), the decision score for class $c$ on an input document is an affine combination of input token features:
            <div style="margin: 0.75rem 0; padding: 0.75rem; background-color: {PALETTE['surface_alt']}; border-left: 3px solid {PALETTE['primary_accent']}; font-family: monospace; font-size: 1.05rem;">
                score(c) = &Sigma;<sub>i &isin; tokens</sub> (x<sub>i</sub> &times; W<sub>c, i</sub>) + b<sub>c</sub>
            </div>
            where:
            <ul>
                <li><b>x<sub>i</sub></b> is the sublinear TF-IDF weight of token <i>i</i> in the input sentence: <code>x<sub>i</sub> = (1 + log(tf<sub>i</sub>)) &times; idf<sub>i</sub></code>.</li>
                <li><b>W<sub>c, i</sub></b> is the model parameter coefficient learned during training for emotion class <i>c</i> and token <i>i</i>.</li>
                <li><b>x<sub>i</sub> &times; W<sub>c, i</sub></b> represents the exact <i>local feature attribution</i>: positive values push the classifier toward emotion <i>c</i>, while negative values penalize it.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# 2. Global Lexicon Top Features per Emotion
st.markdown(f"### {t('explainability.global_importance', locale)}")
st.markdown(
    f"<div style='font-size: 0.88rem; color: {PALETTE['text_secondary']}; margin-bottom: 0.75rem;'>Top weighted n-grams extracted directly from the fitted 12,000-feature Linear SVC weight matrix:</div>",
    unsafe_allow_html=True
)

global_features = get_global_top_features(n_features=12, model_name="Linear SVC")

tabs = st.tabs([f"{EMOTION_LABELS[i].capitalize()}" for i in range(len(EMOTION_LABELS))])

for i, tab in enumerate(tabs):
    emotion = EMOTION_LABELS[i]
    color = EMOTION_COLORS.get(emotion, PALETTE["primary_accent"])
    with tab:
        feats = global_features.get(emotion, [])
        if feats:
            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                df_f = pd.DataFrame(feats)
                df_f.columns = ["Vocabulary Feature (N-Gram)", "Model Weight Coefficient"]
                st.dataframe(df_f, use_container_width=True, hide_index=True)
            with col_t2:
                st.markdown(
                    f"""
                    <div class="editorial-card-alt" style="border-left: 4px solid {color};">
                        <div style="font-weight: 600; color: {color}; font-size: 1rem; margin-bottom: 0.4rem;">
                            Key Lexical Markers for {emotion.capitalize()}
                        </div>
                        <div style="font-size: 0.88rem; line-height: 1.6; color: {PALETTE['text_primary']};">
                            The highest coefficients correspond to explicit affective adjectives (e.g., <i>{feats[0]['feature']}</i>, <i>{feats[1]['feature']}</i>) as well as emotion-reinforcing bigrams. These weights demonstrate that the model successfully learned emotion-specific polarity.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")

# 3. Interactive Local Sentence Attribution Sandbox
st.markdown(f"### {t('explainability.local_attribution', locale)}")
st.markdown(
    f"<div style='font-size: 0.88rem; color: {PALETTE['text_secondary']}; margin-bottom: 0.75rem;'>Test any custom sentence to inspect real-time TF-IDF feature contributions:</div>",
    unsafe_allow_html=True
)

sandbox_text = st.text_input(
    "Sentence to Inspect:",
    value="I feel thoroughly amazed and grateful for this incredible support."
)

col_sb1, col_sb2 = st.columns([1, 1])

with col_sb1:
    target_emotion_name = st.selectbox(
        "Evaluate attributions for emotion class:",
        options=list(EMOTION_LABELS.values()),
        index=1  # Joy by default
    )

target_id = [k for k, v in EMOTION_LABELS.items() if v == target_emotion_name][0]

if sandbox_text.strip():
    local_exp = explain_prediction(
        text=sandbox_text,
        predicted_class_id=target_id,
        model_name="Linear SVC",
        top_k=8
    )

    top_pos = local_exp.get("top_positive", [])
    if top_pos:
        st.markdown(f"#### Top Positive Contributing Tokens for *{target_emotion_name.capitalize()}*")
        fig_c = plot_token_contributions(top_pos)
        if fig_c:
            st.plotly_chart(fig_c, use_container_width=True)

        # Token breakdown table
        st.markdown("##### Detailed Token Weight Decomposition")
        df_tok = pd.DataFrame(top_pos)
        df_tok.columns = ["Token / N-Gram", "TF-IDF Value", "Learned Weight W", "Contribution (x × W)"]
        st.dataframe(df_tok, use_container_width=True, hide_index=True)
    else:
        st.info("No matching vocabulary tokens found in the TF-IDF feature space.")

# Academic Disclaimer Card
st.markdown(
    f"""
    <div class="editorial-card-alt" style="margin-top: 1.5rem; font-size: 0.82rem; color: {PALETTE['text_secondary']};">
        <b>Academic Boundary of Interpretability:</b><br>
        Feature attribution provides transparency into <i>model decision boundaries</i> within the mathematical vector space. A high token weight indicates that the classifier relies on this n-gram to separate emotion hyperplanes; it does not constitute clinical proof of human emotional experience or psychological intent.
    </div>
    """,
    unsafe_allow_html=True
)
