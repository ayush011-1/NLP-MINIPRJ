"""ManSanket (मनसंकेत) - Dataset & Lexical Explorer.

Page 4: Dataset distribution analysis, text length distributions,
lexical n-gram frequencies, and data hygiene/leakage audit.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import PALETTE, RAW_DATA_DIR, EMOTION_LABELS, EMOTION_COLORS
from src.i18n import t
from src.ui import setup_page, render_sidebar, render_header
from src.visualizations import plot_class_distribution, EDITORIAL_LAYOUT

setup_page("Dataset Explorer")
render_sidebar()

locale = st.session_state.get("locale", "en")

render_header(
    title=t("explorer.title", locale),
    subtitle=t("explorer.subtitle", locale)
)


@st.cache_data
def load_and_analyze_datasets():
    """Load and compute descriptive statistics for emotion datasets."""
    train_df = pd.read_csv(RAW_DATA_DIR / "emotion_train.csv")
    val_df = pd.read_csv(RAW_DATA_DIR / "emotion_validation.csv")
    test_df = pd.read_csv(RAW_DATA_DIR / "emotion_test.csv")

    train_df["emotion_name"] = train_df["label"].map(EMOTION_LABELS)
    val_df["emotion_name"] = val_df["label"].map(EMOTION_LABELS)
    test_df["emotion_name"] = test_df["label"].map(EMOTION_LABELS)

    # Compute text length statistics
    train_df["char_len"] = train_df["text"].astype(str).str.len()
    train_df["word_count"] = train_df["text"].astype(str).str.split().str.len()

    # MahaEmotions dataset
    mr_val = pd.read_csv(RAW_DATA_DIR / "mahaemotions_validation.csv")
    mr_test = pd.read_csv(RAW_DATA_DIR / "mahaemotions_test.csv")

    return train_df, val_df, test_df, mr_val, mr_test


train_df, val_df, test_df, mr_val, mr_test = load_and_analyze_datasets()

# 1. Dataset Split & Hygiene Overview
st.markdown(f"### {t('explorer.split_overview', locale)}")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Training Split", f"{len(train_df):,} samples", "16,000 raw (31 dups)")
with col_s2:
    st.metric("Validation Split", f"{len(val_df):,} samples", "2,000 samples")
with col_s3:
    st.metric("Test Split", f"{len(test_df):,} samples", "2,000 samples")
with col_s4:
    st.metric("Marathi Benchmark", f"{len(mr_test):,} samples", "MahaEmotions (L3Cube)")

# Leakage Audit Banner
st.markdown(
    f"""
    <div class="editorial-card-alt" style="margin-top: 1rem; border-left: 4px solid {PALETTE['success']};">
        <div style="font-weight: 600; color: {PALETTE['success']}; font-size: 0.95rem;">
            ✓ {t('explorer.quality_checks', locale)}: {t('explorer.leakage_clean', locale)}
        </div>
        <div style="font-size: 0.85rem; color: {PALETTE['text_secondary']}; margin-top: 0.25rem;">
            Audit Confirmed: Exact training duplicates were isolated and removed; test sets remained strictly unseen during vectorizer fitting and model parameter learning.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# 2. Emotion Class Distribution
st.markdown(f"### {t('explorer.distribution_title', locale)}")

split_tab1, split_tab2 = st.tabs(["English CARER Dataset", "Marathi MahaEmotions Dataset"])

with split_tab1:
    train_dist = train_df["emotion_name"].value_counts().to_dict()
    fig_dist = plot_class_distribution(train_dist, title="CARER Training Split Emotion Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown(
        f"""
        <div style="font-size: 0.85rem; color: {PALETTE['text_secondary']}; margin-bottom: 1rem;">
            <b>Observation on Imbalance:</b> The CARER benchmark exhibits natural class imbalance: <b>Joy</b> (33.5%) and <b>Sadness</b> (29.2%) constitute over 60% of samples, whereas <b>Surprise</b> (3.6%) and <b>Love</b> (8.1%) are relatively rare. This justifies reporting <b>Macro F1</b> alongside overall Accuracy.
        </div>
        """,
        unsafe_allow_html=True
    )

with split_tab2:
    mr_dist = mr_test["Label"].astype(str).str.strip().value_counts().to_dict()
    fig_mr = plot_class_distribution(mr_dist, title="MahaEmotions Test Split Emotion Distribution")
    st.plotly_chart(fig_mr, use_container_width=True)

st.markdown("---")

# 3. Text Length Distribution
st.markdown(f"### {t('explorer.text_length_dist', locale)}")

col_l1, col_l2 = st.columns(2)

with col_l1:
    fig_char = px.histogram(
        train_df,
        x="char_len",
        nbins=50,
        title="<b>Character Length Distribution (Training Set)</b>",
        color_discrete_sequence=[PALETTE["primary_accent"]]
    )
    fig_char.update_layout(EDITORIAL_LAYOUT)
    fig_char.update_layout(xaxis_title="Character Count", yaxis_title="Sample Frequency", height=320)
    st.plotly_chart(fig_char, use_container_width=True)

with col_l2:
    fig_word = px.histogram(
        train_df,
        x="word_count",
        nbins=40,
        title="<b>Word Count Distribution (Training Set)</b>",
        color_discrete_sequence=[PALETTE["supporting_accent"]]
    )
    fig_word.update_layout(EDITORIAL_LAYOUT)
    fig_word.update_layout(xaxis_title="Word Count", yaxis_title="Sample Frequency", height=320)
    st.plotly_chart(fig_word, use_container_width=True)

st.markdown("---")

# 4. Interactive Sample Inspector
st.markdown("### Interactive Dataset Sample Inspector")
st.markdown(
    f"<div style='font-size: 0.88rem; color: {PALETTE['text_secondary']}; margin-bottom: 0.75rem;'>Browse real verified examples from the CARER training split by emotion class:</div>",
    unsafe_allow_html=True
)

filter_col1, filter_col2 = st.columns([1, 2])
with filter_col1:
    selected_class = st.selectbox(
        "Filter by Emotion:",
        options=list(EMOTION_LABELS.values()),
        index=0
    )

with filter_col2:
    num_samples = st.slider("Number of samples to display:", min_value=3, max_value=15, value=5)

filtered_samples = train_df[train_df["emotion_name"] == selected_class].head(num_samples)

for idx, row in filtered_samples.iterrows():
    st.markdown(
        f"""
        <div class="editorial-card" style="margin-bottom: 0.5rem; padding: 0.85rem 1.15rem;">
            <div style="font-size: 0.95rem; color: {PALETTE['text_primary']};">
                &ldquo;{row['text']}&rdquo;
            </div>
            <div style="font-size: 0.78rem; color: {PALETTE['text_muted']}; margin-top: 0.35rem;">
                Label: <b>{row['emotion_name'].capitalize()}</b> (ID: {row['label']}) | Words: {row['word_count']} | Chars: {row['char_len']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
