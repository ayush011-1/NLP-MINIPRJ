"""Explainability and feature attribution module for ManSanket (मनसंकेत).

Implements:
1. Linear model feature attribution (TF-IDF value * weight coefficient).
2. Local sentence-level token importance scoring and visual highlights.
3. Global vocabulary feature rankings per emotion class.
4. Scientific disclaimers and limitations of linear feature attribution.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.config import EMOTION_LABELS, MODELS_DIR
from src.preprocessing import clean_text, tokenize
import joblib

_RAW_LINEAR_SVC = None
_LOGISTIC_REGRESSION = None
_VECTORIZER = None


def get_models_for_explainability():
    """Lazily load models needed for coefficient attribution."""
    global _RAW_LINEAR_SVC, _LOGISTIC_REGRESSION, _VECTORIZER
    if _RAW_LINEAR_SVC is None:
        raw_svc_path = MODELS_DIR / "raw_linear_svc.joblib"
        if raw_svc_path.exists():
            _RAW_LINEAR_SVC = joblib.load(raw_svc_path)
    if _LOGISTIC_REGRESSION is None:
        lr_path = MODELS_DIR / "logistic_regression.joblib"
        if lr_path.exists():
            _LOGISTIC_REGRESSION = joblib.load(lr_path)
    if _VECTORIZER is None:
        vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
        if vec_path.exists():
            _VECTORIZER = joblib.load(vec_path)
    return _RAW_LINEAR_SVC, _LOGISTIC_REGRESSION, _VECTORIZER


def explain_prediction(
    text: str,
    predicted_class_id: int,
    model_name: str = "Linear SVC",
    top_k: int = 10
) -> Dict[str, Any]:
    """Compute local token-level feature attribution for a given prediction.
    
    Args:
        text: Input string.
        predicted_class_id: The integer ID of the predicted emotion.
        model_name: 'Linear SVC' or 'Logistic Regression'.
        top_k: Number of top influential terms to return.
        
    Returns:
        Dict containing top positive tokens, negative tokens, token highlights, and explanation note.
    """
    raw_svc, lr, vectorizer = get_models_for_explainability()
    
    clf = raw_svc if "SVC" in model_name and raw_svc is not None else lr
    if clf is None or vectorizer is None:
        return {
            "top_positive": [],
            "top_negative": [],
            "token_attributions": [],
            "attribution_available": False,
            "note": "Fitted model coefficients are currently unavailable for attribution."
        }

    # Transform input text
    cleaned = clean_text(text)
    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_vec = vectorizer.transform([cleaned])
    
    # Extract non-zero indices in input
    non_zero_indices = tfidf_vec.nonzero()[1]
    if len(non_zero_indices) == 0:
        return {
            "top_positive": [],
            "top_negative": [],
            "token_attributions": [],
            "attribution_available": False,
            "note": "No vocabulary tokens matched the TF-IDF feature space."
        }

    # Extract weights for the predicted class
    coef = clf.coef_[predicted_class_id]  # Shape: (n_features,)
    
    contributions = []
    for idx in non_zero_indices:
        tfidf_val = tfidf_vec[0, idx]
        weight = coef[idx]
        contrib = float(tfidf_val * weight)
        feature_name = feature_names[idx]
        contributions.append({
            "feature": feature_name,
            "tfidf": round(float(tfidf_val), 4),
            "weight": round(float(weight), 4),
            "contribution": round(contrib, 4)
        })

    # Sort contributions
    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    top_positive = [c for c in contributions if c["contribution"] > 0][:top_k]
    top_negative = [c for c in sorted(contributions, key=lambda x: x["contribution"]) if c["contribution"] < 0][:top_k]

    # Compute sentence-level token highlights
    tokens = tokenize(cleaned, lower=True)
    contrib_map = {c["feature"]: c["contribution"] for c in contributions}
    
    # Normalize contributions for coloring (scale from -1 to 1)
    max_abs = max([abs(c["contribution"]) for c in contributions] or [1.0])
    if max_abs == 0:
        max_abs = 1.0

    token_attributions = []
    for t in tokens:
        score = contrib_map.get(t, 0.0)
        norm_score = score / max_abs
        token_attributions.append({
            "token": t,
            "raw_contribution": round(score, 4),
            "normalized_score": round(norm_score, 3),
            "is_influential": abs(score) > 0.01
        })

    pred_emotion = EMOTION_LABELS.get(predicted_class_id, "unknown")
    explanation_note = (
        f"For the predicted emotion '{pred_emotion}', linear coefficients identify tokens whose "
        f"TF-IDF representation exerted the strongest positive shift on the classifier's decision boundary."
    )

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "token_attributions": token_attributions,
        "attribution_available": True,
        "note": explanation_note,
        "predicted_emotion": pred_emotion
    }


def get_global_top_features(n_features: int = 15, model_name: str = "Linear SVC") -> Dict[str, List[Dict[str, Any]]]:
    """Extract top positive vocabulary n-grams for every emotion class."""
    raw_svc, lr, vectorizer = get_models_for_explainability()
    clf = raw_svc if "SVC" in model_name and raw_svc is not None else lr
    
    if clf is None or vectorizer is None:
        return {}

    feature_names = np.array(vectorizer.get_feature_names_out())
    results = {}

    for class_id, emotion in EMOTION_LABELS.items():
        if class_id >= len(clf.coef_):
            continue
        coefs = clf.coef_[class_id]
        top_indices = np.argsort(coefs)[-n_features:][::-1]
        
        features_list = []
        for idx in top_indices:
            features_list.append({
                "feature": feature_names[idx],
                "coefficient": round(float(coefs[idx]), 4)
            })
        results[emotion] = features_list

    return results
