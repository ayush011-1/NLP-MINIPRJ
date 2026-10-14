"""ManSanket (मनसंकेत) - Methodology, Mathematical Foundations & Viva Guide.

Page 6: Comprehensive academic report covering NLP pipeline stages,
mathematical formulas for all classifiers, and a viva examination preparation guide.
"""
import streamlit as st
from src.config import PALETTE
from src.i18n import t
from src.ui import setup_page, render_sidebar, render_header

setup_page("Methodology & Viva")
render_sidebar()

locale = st.session_state.get("locale", "en")

render_header(
    title=t("methodology.title", locale),
    subtitle=t("methodology.subtitle", locale)
)

tabs = st.tabs([
    "1. Project Overview & Architecture",
    "2. NLP Preprocessing & Feature Extraction",
    "3. Mathematical Formulations",
    "4. Viva Questions & Examination Guide"
])

# Tab 1: Project Overview & Architecture
with tabs[0]:
    st.markdown(
        f"""
        <div class="editorial-card">
            <h3 style="color: {PALETTE['primary_accent']}; margin-top: 0;">1. Problem Statement & Motivation</h3>
            <p style="font-size: 0.92rem; line-height: 1.6; color: {PALETTE['text_primary']};">
                Textual emotion detection aims to identify fine-grained human emotional states from unstructured written language. While conventional sentiment analysis reduces text to positive, negative, or neutral polarity, emotional analysis distinguishes nuanced psychological categories such as joy, sadness, fear, and anger. This distinction is critical in human-computer interaction, mental well-being monitoring, and social media analytics.
            </p>
            <h4 style="color: {PALETTE['supporting_accent']}; margin-top: 1rem;">Project Objectives</h4>
            <ul style="font-size: 0.9rem; line-height: 1.6; color: {PALETTE['text_primary']};">
                <li>Implement an end-to-end NLP classification pipeline using the standardized 6-class CARER emotion taxonomy.</li>
                <li>Compare three traditional machine learning classifiers: <b>Multinomial Naive Bayes</b>, <b>Logistic Regression</b>, and <b>Linear Support Vector Classification (Linear SVC)</b> with calibrated probabilities.</li>
                <li>Incorporate script and language awareness (English, Hindi, Marathi, Hinglish) with transparent distinction between native and translation-mediated inference.</li>
                <li>Provide local and global feature explainability using linear decision coefficients.</li>
                <li>Ensure academic reproducibility with zero-leakage benchmark evaluation.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="editorial-card-alt">
            <h4 style="color: {PALETTE['primary_accent']}; margin-top: 0;">System Architecture Flow</h4>
            <div style="font-size: 0.88rem; line-height: 1.6; color: {PALETTE['text_secondary']};">
                Raw Input Text &rarr; Unicode NFKC Normalization &rarr; Sanitization (URLs, emails, handles) &rarr; Contraction Expansion (preserving negations) &rarr; Language & Script Analysis &rarr; Sublinear TF-IDF Vectorization (unigrams + bigrams) &rarr; Calibrated Linear Classifier &rarr; Class Probabilities & Linear Token Attributions &rarr; Editorial Interface.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Tab 2: NLP Preprocessing & Feature Extraction
with tabs[1]:
    st.markdown(
        f"""
        <div class="editorial-card">
            <h3 style="color: {PALETTE['primary_accent']}; margin-top: 0;">2. Preprocessing & TF-IDF Vectorization</h3>
            <p style="font-size: 0.92rem; line-height: 1.6; color: {PALETTE['text_primary']};">
                Proper preprocessing is crucial in emotion detection because aggressive stop-word removal can inadvertently erase emotional valence.
            </p>
            
            <h4 style="color: {PALETTE['supporting_accent']};">A. Unicode NFKC Normalization</h4>
            <p style="font-size: 0.9rem; color: {PALETTE['text_primary']};">
                Both Latin and Devanagari scripts (Hindi, Marathi) contain composite characters and combining diacritics (matras). Normalization via <code>unicodedata.normalize('NFKC', text)</code> ensures consistent canonical glyph representation across different keyboards and input devices.
            </p>

            <h4 style="color: {PALETTE['supporting_accent']};">B. Contraction Expansion with Negation Preservation</h4>
            <p style="font-size: 0.9rem; color: {PALETTE['text_primary']};">
                English contractions such as <i>"didn't"</i>, <i>"can't"</i>, and <i>"wasn't"</i> are expanded to <i>"did not"</i>, <i>"cannot"</i>, and <i>"was not"</i>. Crucially, negations (<i>not, never, no</i>) are <b>strictly retained</b> because removing them reverses the emotional polarity (e.g., "not happy" vs "happy").
            </p>

            <h4 style="color: {PALETTE['supporting_accent']};">C. Sublinear Term Frequency-Inverse Document Frequency (TF-IDF)</h4>
            <p style="font-size: 0.9rem; color: {PALETTE['text_primary']};">
                Text features are extracted using unigram and bigram token sequences (n-gram range 1 to 2) with a maximum vocabulary of 12,000 features. We use <b>sublinear TF scaling</b>:
            </p>
            <div style="padding: 0.65rem; background-color: {PALETTE['surface_alt']}; font-family: monospace; font-size: 1rem; border-left: 3px solid {PALETTE['primary_accent']};">
                TF<sub>sublinear</sub>(t, d) = 1 + log(TF(t, d)) &nbsp;&nbsp;(for TF > 0)<br>
                IDF(t, D) = log((1 + |D|) / (1 + DF(t))) + 1<br>
                TF-IDF(t, d, D) = TF<sub>sublinear</sub>(t, d) &times; IDF(t, D)
            </div>
            <p style="font-size: 0.88rem; color: {PALETTE['text_secondary']}; margin-top: 0.5rem;">
                Sublinear scaling dampens the effect of terms appearing excessively in a document (a term appearing 20 times is not 20 times more important than one appearing once).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Tab 3: Mathematical Formulations
with tabs[2]:
    st.markdown(
        f"""
        <div class="editorial-card">
            <h3 style="color: {PALETTE['primary_accent']}; margin-top: 0;">3. Mathematical Foundations of Classifiers</h3>
            
            <h4 style="color: {PALETTE['supporting_accent']};">A. Multinomial Naive Bayes</h4>
            <p style="font-size: 0.9rem; line-height: 1.6; color: {PALETTE['text_primary']};">
                Based on Bayes' theorem with the conditional independence assumption between features given the class:
            </p>
            <div style="padding: 0.65rem; background-color: {PALETTE['surface_alt']}; font-family: monospace; font-size: 0.95rem; border-left: 3px solid {PALETTE['primary_accent']};">
                P(c | x) &prop; P(c) &prod;<sub>i</sub> P(x<sub>i</sub> | c)<br>
                P(x<sub>i</sub> | c) = (N<sub>ci</sub> + &alpha;) / (N<sub>c</sub> + &alpha; &times; |V|)
            </div>
            <p style="font-size: 0.88rem; color: {PALETTE['text_secondary']}; margin-top: 0.35rem;">
                Laplace smoothing (&alpha; = 0.5) prevents zero-probability errors on unseen terms.
            </p>

            <h4 style="color: {PALETTE['supporting_accent']}; margin-top: 1rem;">B. Multinomial Logistic Regression</h4>
            <p style="font-size: 0.9rem; line-height: 1.6; color: {PALETTE['text_primary']};">
                Models posterior probabilities using the Softmax (normalized exponential) function over linear combinations:
            </p>
            <div style="padding: 0.65rem; background-color: {PALETTE['surface_alt']}; font-family: monospace; font-size: 0.95rem; border-left: 3px solid {PALETTE['primary_accent']};">
                P(y = c | x) = exp(w<sub>c</sub><sup>T</sup> x + b<sub>c</sub>) / &sum;<sub>k=1..K</sub> exp(w<sub>k</sub><sup>T</sup> x + b<sub>k</sub>)
            </div>
            <p style="font-size: 0.88rem; color: {PALETTE['text_secondary']}; margin-top: 0.35rem;">
                Trained using cross-entropy loss with L2 regularization (C = 2.0).
            </p>

            <h4 style="color: {PALETTE['supporting_accent']}; margin-top: 1rem;">C. Linear Support Vector Classification (Linear SVC)</h4>
            <p style="font-size: 0.9rem; line-height: 1.6; color: {PALETTE['text_primary']};">
                Finds the optimal separating hyperplane that maximizes the geometric margin between emotion classes:
            </p>
            <div style="padding: 0.65rem; background-color: {PALETTE['surface_alt']}; font-family: monospace; font-size: 0.95rem; border-left: 3px solid {PALETTE['primary_accent']};">
                min<sub>w, b</sub> (1/2 ||w||<sup>2</sup> + C &sum;<sub>j</sub> max(0, 1 - y<sub>j</sub>(w<sup>T</sup> x<sub>j</sub> + b)))
            </div>
            <p style="font-size: 0.88rem; color: {PALETTE['text_secondary']}; margin-top: 0.35rem;">
                Linear SVC achieves the highest accuracy (88.25%) on the test benchmark. CalibratedClassifierCV with 3-fold cross-validation applies Platt sigmoid scaling to convert geometric margin distances into calibrated class probabilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Tab 4: Viva Questions & Examination Guide
with tabs[3]:
    st.markdown(
        f"""
        <div class="editorial-card">
            <h3 style="color: {PALETTE['primary_accent']}; margin-top: 0;">4. Viva Examination Preparation Guide</h3>
            <p style="font-size: 0.9rem; color: {PALETTE['text_secondary']}; margin-bottom: 1rem;">
                Key conceptual and practical questions frequently asked during mini-project viva evaluations:
            </p>

            <div style="margin-bottom: 1rem; padding: 0.75rem; background-color: {PALETTE['surface_alt']}; border-radius: 4px;">
                <b>Q1. What is the fundamental difference between Sentiment Analysis and Emotion Detection?</b><br>
                <i>Answer:</i> Sentiment analysis identifies coarse polarity (positive, negative, or neutral). Emotion detection identifies specific affective states (joy, sadness, anger, fear, surprise, love). Both fear and anger have negative sentiment polarity, but represent completely distinct cognitive reactions and behavioural intents.
            </div>

            <div style="margin-bottom: 1rem; padding: 0.75rem; background-color: {PALETTE['surface_alt']}; border-radius: 4px;">
                <b>Q2. Why is Macro F1 preferred over overall Accuracy on this dataset?</b><br>
                <i>Answer:</i> The dataset exhibits class imbalance (Joy: 33.5%, Surprise: 3.6%). Overall accuracy is dominated by the majority classes. Macro F1 calculates the unweighted mean of F1-scores across all classes, treating rare classes (such as surprise) with equal importance and penalizing models that ignore minority classes.
            </div>

            <div style="margin-bottom: 1rem; padding: 0.75rem; background-color: {PALETTE['surface_alt']}; border-radius: 4px;">
                <b>Q3. Why did Linear SVC outperform Naive Bayes?</b><br>
                <i>Answer:</i> Naive Bayes makes the strong assumption of conditional independence between features given the class. In natural text, word co-occurrences and bi-grams violate this assumption. Linear SVC maximizes the geometric margin without independence assumptions and effectively handles high-dimensional, sparse TF-IDF vectors.
            </div>

            <div style="margin-bottom: 1rem; padding: 0.75rem; background-color: {PALETTE['surface_alt']}; border-radius: 4px;">
                <b>Q4. How does the multilingual pipeline handle Devanagari script?</b><br>
                <i>Answer:</i> The system uses Unicode NFKC normalization and script analysis. Because both Hindi and Marathi use the Devanagari script, the system checks for Marathi-distinctive phonemes (like ळ) and grammatical markers (आहे, नाही, मला vs है, नहीं, मुझे) to disambiguate the language.
            </div>

            <div style="margin-bottom: 1rem; padding: 0.75rem; background-color: {PALETTE['surface_alt']}; border-radius: 4px;">
                <b>Q5. How is explainability achieved in your system?</b><br>
                <i>Answer:</i> In linear models, the decision score is &Sigma; x<sub>i</sub> &times; W<sub>c, i</sub> + b<sub>c</sub>. By multiplying the TF-IDF feature value x<sub>i</sub> of each token in the sentence by the learned class coefficient W<sub>c, i</sub>, we compute exact local feature attributions that reveal which tokens shifted the boundary toward the predicted emotion.
            </div>

            <div style="padding: 0.75rem; background-color: {PALETTE['surface_alt']}; border-radius: 4px;">
                <b>Q6. Why did direct zero-shot evaluation on Marathi MahaEmotions yield ~30% accuracy?</b><br>
                <i>Answer:</i> Because the TF-IDF feature space was constructed from English vocabulary. Native Devanagari words produce zero overlap with the English vocabulary matrix (out-of-vocabulary), causing the model to default to the majority class prior. This is why multilingual subword embeddings or translation mediation are required.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
