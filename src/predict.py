"""Prediction and inference pipeline for ManSanket (मनसंकेत).

Handles:
- Language detection and routing
- Optional user-consented translation mediation for non-English input
- Model inference with calibrated class probability distributions
- Local token feature attribution integration
- Detailed linguistic summary metrics
"""
import joblib
import numpy as np
from typing import Dict, Any, Optional
from src.config import (
    VECTORIZER_PATH,
    MULTINOMIAL_NB_PATH,
    LOGISTIC_REGRESSION_PATH,
    LINEAR_SVC_PATH,
    MODEL_METADATA_PATH,
    EMOTION_LABELS,
    LABEL_TO_ID
)
from src.preprocessing import clean_text, compute_text_statistics
from src.language_detection import detect_language
from src.explainability import explain_prediction

# Model cache
_MODELS_CACHE = {}


def load_inference_models():
    """Load and cache vectorizer and trained classifiers."""
    global _MODELS_CACHE
    if _MODELS_CACHE:
        return _MODELS_CACHE

    if VECTORIZER_PATH.exists():
        _MODELS_CACHE["vectorizer"] = joblib.load(VECTORIZER_PATH)
    if LINEAR_SVC_PATH.exists():
        _MODELS_CACHE["Linear SVC"] = joblib.load(LINEAR_SVC_PATH)
    if LOGISTIC_REGRESSION_PATH.exists():
        _MODELS_CACHE["Logistic Regression"] = joblib.load(LOGISTIC_REGRESSION_PATH)
    if MULTINOMIAL_NB_PATH.exists():
        _MODELS_CACHE["Multinomial Naive Bayes"] = joblib.load(MULTINOMIAL_NB_PATH)

    return _MODELS_CACHE


def translate_with_consent(text: str, source_lang: str) -> Dict[str, Any]:
    """Attempt translation to English with safe exception and rate-limit handling.
    
    Sanitizes URLs, email addresses, and personal handles before sending
    to the external translation service.
    
    Args:
        text: Input string.
        source_lang: Detected source language code ('hi', 'mr', etc.).
        
    Returns:
        Dict with success status, translated text, and explanatory notes.
    """
    try:
        from deep_translator import GoogleTranslator
        # Sanitize sensitive patterns (URLs, emails, handles) prior to translation
        sanitized_input = clean_text(text, expand_contr=False)
        if not sanitized_input:
            sanitized_input = text

        src = "mr" if source_lang in ["mr", "mr-en-codemixed"] else ("hi" if source_lang in ["hi", "hinglish"] else "auto")
        translator = GoogleTranslator(source=src, target="en")
        translated = translator.translate(sanitized_input)
        return {
            "success": True,
            "translated_text": translated,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "translated_text": text,
            "error": f"The external translation service could not be reached ({type(e).__name__}). Analyzed original text directly."
        }


def predict_emotion(
    text: str,
    model_name: str = "Linear SVC",
    manual_lang: Optional[str] = None,
    allow_translation: bool = False
) -> Dict[str, Any]:
    """Execute end-to-end emotion prediction and analysis on user text.
    
    Args:
        text: Input text string.
        model_name: Classifier name ('Linear SVC', 'Logistic Regression', 'Multinomial Naive Bayes').
        manual_lang: Manual language override if user selected one.
        allow_translation: Whether the user explicitly enabled translation mediation.
        
    Returns:
        Structured dictionary with predictions, probabilities, attributions, and statistics.
    """
    if not text or not text.strip():
        return {
            "is_valid": False,
            "error": "Empty input provided.",
            "text": ""
        }

    models = load_inference_models()
    vectorizer = models.get("vectorizer")
    clf = models.get(model_name) or models.get("Linear SVC")

    if vectorizer is None or clf is None:
        return {
            "is_valid": False,
            "error": "Trained model or TF-IDF vectorizer artifacts not found. Please train models first.",
            "text": text
        }

    # 1. Language Detection & Verification
    lang_info = detect_language(text)
    effective_lang = manual_lang if (manual_lang and manual_lang != "auto") else lang_info["language_code"]

    # 2. Determine Route & Translation Mediation
    is_translation_mediated = False
    translated_text = None
    translation_note = None

    text_to_analyze = text

    # Handle unsupported languages
    if not lang_info["is_supported"] and effective_lang == "unsupported":
        route_type = "unsupported"
        route_message = f"Language ({lang_info['language_name']}) is not directly supported by ManSanket."
    elif effective_lang in ["hi", "mr", "hinglish", "mr-en-codemixed", "code-mixed"]:
        if allow_translation:
            trans_result = translate_with_consent(text, effective_lang)
            if trans_result["success"]:
                translated_text = trans_result["translated_text"]
                text_to_analyze = translated_text
                is_translation_mediated = True
                route_type = "translation-mediated"
                route_message = "Translation-mediated inference: Input was translated to English for evaluation against the English benchmark model."
            else:
                route_type = "experimental"
                translation_note = trans_result["error"]
                route_message = "Direct evaluation on non-English text without translation (experimental route; vocabulary coverage may be limited)."
        else:
            route_type = "experimental"
            route_message = "Direct non-English inference without translation mediation. High risk of out-of-vocabulary terms on English-trained TF-IDF model."
    else:
        route_type = "validated"
        route_message = "Direct validated model inference on English text."

    # 3. Preprocess and Vectorize
    cleaned = clean_text(text_to_analyze)
    stats = compute_text_statistics(text)
    
    # 4. Predict
    tfidf_vec = vectorizer.transform([cleaned])
    
    # Get calibrated probabilities
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(tfidf_vec)[0]
    else:
        # Fallback to decision function softmax if raw LinearSVC
        decision = clf.decision_function(tfidf_vec)[0]
        exp_d = np.exp(decision - np.max(decision))
        probs = exp_d / np.sum(exp_d)

    predicted_id = int(np.argmax(probs))
    confidence = float(probs[predicted_id])
    predicted_label = EMOTION_LABELS.get(predicted_id, "unknown")

    prob_dict = {
        EMOTION_LABELS[i]: round(float(probs[i]), 4)
        for i in range(len(EMOTION_LABELS))
    }

    # 5. Explainability (Feature Attribution)
    # Explain on text_to_analyze (which is translated text if translation was used)
    explain_res = explain_prediction(
        text=text_to_analyze,
        predicted_class_id=predicted_id,
        model_name=model_name,
        top_k=8
    )

    return {
        "is_valid": True,
        "original_text": text,
        "analyzed_text": text_to_analyze,
        "cleaned_text": cleaned,
        "translated_text": translated_text,
        "is_translation_mediated": is_translation_mediated,
        "translation_note": translation_note,
        "language_info": lang_info,
        "effective_lang": effective_lang,
        "model_name": model_name,
        "predicted_id": predicted_id,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "route_type": route_type,
        "route_message": route_message,
        "statistics": stats,
        "top_features": explain_res.get("top_positive", []),
        "top_negative_features": explain_res.get("top_negative", []),
        "token_attributions": explain_res.get("token_attributions", []),
        "attribution_note": explain_res.get("note", "")
    }
