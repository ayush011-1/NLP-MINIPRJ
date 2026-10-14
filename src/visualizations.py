"""Editorial visualization utilities for ManSanket (मनसंकेत).

Generates custom-themed Plotly charts adhering to the editorial palette:
- Warm paper background (#F7F5F0)
- Deep Slate Blue (#35465C) & Terracotta (#A45D48)
- Curated emotion-specific colors
- Clean typography and subdued grid lines
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union

from src.config import PALETTE, EMOTION_COLORS, EMOTION_LABELS

# Shared editorial layout template
EDITORIAL_LAYOUT = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["surface"],
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color=PALETTE["text_primary"]),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(
        gridcolor=PALETTE["border"],
        zerolinecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["text_secondary"]),
        title_font=dict(color=PALETTE["text_primary"])
    ),
    yaxis=dict(
        gridcolor=PALETTE["border"],
        zerolinecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["text_secondary"]),
        title_font=dict(color=PALETTE["text_primary"])
    )
)


def plot_probability_bars(probabilities: Dict[str, float]) -> go.Figure:
    """Create a styled horizontal bar chart showing emotion class probabilities."""
    emotions = list(probabilities.keys())
    scores = [probabilities[e] for e in emotions]
    colors = [EMOTION_COLORS.get(e, PALETTE["primary_accent"]) for e in emotions]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=[e.capitalize() for e in emotions],
            orientation="h",
            marker=dict(color=colors, line=dict(color=PALETTE["border"], width=1)),
            text=[f"{s*100:.1f}%" for s in scores],
            textposition="outside",
            cliponaxis=False
        )
    )

    layout = EDITORIAL_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Emotion Probability Distribution</b>", font=dict(size=15, color=PALETTE["primary_accent"])),
        xaxis=dict(range=[0, 1.15], tickformat=".0%", showgrid=True, gridcolor=PALETTE["border"]),
        yaxis=dict(autorange="reversed"),
        height=320
    )
    fig.update_layout(layout)
    return fig


def plot_token_contributions(top_features: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Create a horizontal bar chart displaying top contributing features."""
    if not top_features:
        return None

    features = [f["feature"] for f in top_features][::-1]
    contribs = [f["contribution"] for f in top_features][::-1]
    
    # Sage teal for positive, Terracotta for negative
    bar_colors = [PALETTE["success"] if c >= 0 else PALETTE["danger"] for c in contribs]

    fig = go.Figure(
        go.Bar(
            x=contribs,
            y=features,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color=PALETTE["border"], width=1)),
            text=[f"{c:+.3f}" for c in contribs],
            textposition="outside",
            cliponaxis=False
        )
    )

    layout = EDITORIAL_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Top Influential Token Attributions (Weight × TF-IDF)</b>", font=dict(size=15, color=PALETTE["primary_accent"])),
        xaxis=dict(title="Feature Contribution Score", showgrid=True, gridcolor=PALETTE["border"]),
        yaxis=dict(title="Token"),
        height=max(260, len(features) * 32)
    )
    fig.update_layout(layout)
    return fig


def plot_confusion_matrix(
    matrix: List[List[Union[int, float]]],
    labels: List[str],
    normalize: bool = False,
    model_name: str = "Linear SVC"
) -> go.Figure:
    """Plot an editorial styled confusion matrix heatmap."""
    z_vals = np.array(matrix)
    text_vals = [[f"{val:.2%}" if normalize else f"{int(val):,}" for val in row] for row in z_vals]

    # Elegant slate blue color scale
    custom_colorscale = [
        [0.0, "#F7F5F0"],
        [0.2, "#D9E2EC"],
        [0.5, "#829AB1"],
        [0.8, "#486581"],
        [1.0, "#102A43"]
    ]

    title_text = f"<b>{model_name} Confusion Matrix ({'Normalized' if normalize else 'Counts'})</b>"
    
    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=[l.capitalize() for l in labels],
            y=[l.capitalize() for l in labels],
            colorscale=custom_colorscale,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=12, color=PALETTE["text_primary"]),
            hoverongaps=False,
            colorbar=dict(title="Ratio" if normalize else "Count", outlinecolor=PALETTE["border"])
        )
    )

    layout = EDITORIAL_LAYOUT.copy()
    layout.update(
        title=dict(text=title_text, font=dict(size=15, color=PALETTE["primary_accent"])),
        xaxis=dict(title="Predicted Emotion", tickfont=dict(size=11)),
        yaxis=dict(title="True Emotion", autorange="reversed", tickfont=dict(size=11)),
        height=450
    )
    fig.update_layout(layout)
    return fig


def plot_model_comparison(evaluation_metrics: Dict[str, Any], split: str = "test") -> go.Figure:
    """Plot comparative bar chart across models on Accuracy, Macro F1, and Weighted F1."""
    models = list(evaluation_metrics.keys())
    accuracies = [evaluation_metrics[m][split]["accuracy"] for m in models]
    macro_f1s = [evaluation_metrics[m][split]["macro_f1"] for m in models]
    weighted_f1s = [evaluation_metrics[m][split]["weighted_f1"] for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy",
        x=models,
        y=accuracies,
        marker=dict(color=PALETTE["primary_accent"]),
        text=[f"{a*100:.1f}%" for a in accuracies],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Macro F1",
        x=models,
        y=macro_f1s,
        marker=dict(color=PALETTE["supporting_accent"]),
        text=[f"{f*100:.1f}%" for f in macro_f1s],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Weighted F1",
        x=models,
        y=weighted_f1s,
        marker=dict(color="#6B7280"),
        text=[f"{w*100:.1f}%" for w in weighted_f1s],
        textposition="outside"
    ))

    layout = EDITORIAL_LAYOUT.copy()
    layout.update(
        title=dict(text=f"<b>Algorithm Benchmark Comparison ({split.capitalize()} Set)</b>", font=dict(size=15, color=PALETTE["primary_accent"])),
        barmode="group",
        yaxis=dict(range=[0, 1.15], tickformat=".0%", title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380
    )
    fig.update_layout(layout)
    return fig


def plot_class_distribution(distribution: Dict[str, int], title: str = "Emotion Class Distribution") -> go.Figure:
    """Plot emotion class frequency bar chart."""
    classes = list(distribution.keys())
    counts = [distribution[c] for c in classes]
    colors = [EMOTION_COLORS.get(c, PALETTE["primary_accent"]) for c in classes]

    fig = go.Figure(
        go.Bar(
            x=[c.capitalize() for c in classes],
            y=counts,
            marker=dict(color=colors, line=dict(color=PALETTE["border"], width=1)),
            text=[f"{cnt:,}" for cnt in counts],
            textposition="outside",
            cliponaxis=False
        )
    )

    layout = EDITORIAL_LAYOUT.copy()
    layout.update(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=PALETTE["primary_accent"])),
        xaxis=dict(title="Emotion Category"),
        yaxis=dict(title="Sample Count"),
        height=360
    )
    fig.update_layout(layout)
    return fig
