"""Model training and evaluation pipeline for ManSanket (मनसंकेत).

Trains and benchmarks:
1. Multinomial Naive Bayes with TF-IDF
2. Logistic Regression with TF-IDF
3. Linear Support Vector Classifier (with probability calibration) with TF-IDF

Performs:
- Rigorous data leakage and duplicate audits
- Evaluation on Validation and Test splits
- Cross-lingual benchmark evaluation on Marathi MahaEmotions
- Persistence of models, vectorizers, metadata, and evaluation metrics
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    EMOTION_LABELS,
    LABEL_TO_ID,
    VECTORIZER_PATH,
    MULTINOMIAL_NB_PATH,
    LOGISTIC_REGRESSION_PATH,
    LINEAR_SVC_PATH,
    MODEL_METADATA_PATH,
    EVALUATION_METRICS_PATH,
    CONFUSION_MATRICES_PATH,
    MULTILINGUAL_RESULTS_PATH
)
from src.preprocessing import clean_text, tokenize
from src.evaluate import compute_metrics

# Mapping from MahaEmotions labels to 6-class CARER taxonomy
MAHAEMOTIONS_MAPPING = {
    'Happiness': 'joy',
    'Happiness ': 'joy',
    'Excitement': 'joy',
    'Sadness': 'sadness',
    'sadness': 'sadness',
    'Anger': 'anger',
    'Disgust': 'anger',
    'Fear': 'fear',
    'Surprise': 'surprise'
}


def load_datasets():
    """Load raw datasets and perform leakage/duplicate checks."""
    print("Loading raw emotion datasets...")
    train_df = pd.read_csv(RAW_DATA_DIR / "emotion_train.csv")
    val_df = pd.read_csv(RAW_DATA_DIR / "emotion_validation.csv")
    test_df = pd.read_csv(RAW_DATA_DIR / "emotion_test.csv")

    # Data hygiene checks
    train_dups = int(train_df['text'].duplicated().sum())
    val_dups = int(val_df['text'].duplicated().sum())
    test_dups = int(test_df['text'].duplicated().sum())

    train_texts = set(train_df['text'].str.strip())
    val_leakage = int(val_df['text'].str.strip().isin(train_texts).sum())
    test_leakage = int(test_df['text'].str.strip().isin(train_texts).sum())

    print(f"Dataset Audit:")
    print(f"  Train samples: {len(train_df)} (duplicates: {train_dups})")
    print(f"  Val samples:   {len(val_df)} (duplicates: {val_dups}, leak into train: {val_leakage})")
    print(f"  Test samples:  {len(test_df)} (duplicates: {test_dups}, leak into train: {test_leakage})")

    # Clean duplicates in train
    train_df = train_df.drop_duplicates(subset=['text']).reset_index(drop=True)

    # Save cleaned processed datasets
    train_df.to_csv(PROCESSED_DATA_DIR / "train_cleaned.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val_cleaned.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_cleaned.csv", index=False)

    return train_df, val_df, test_df, {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_duplicates_removed": train_dups,
        "val_leakage": val_leakage,
        "test_leakage": test_leakage
    }


def load_marathi_benchmark():
    """Load and map Marathi MahaEmotions benchmark."""
    val_path = RAW_DATA_DIR / "mahaemotions_validation.csv"
    test_path = RAW_DATA_DIR / "mahaemotions_test.csv"

    results = {}
    for name, path in [("validation", val_path), ("test", test_path)]:
        if path.exists():
            df = pd.read_csv(path)
            # Find tweet and label columns
            tweet_col = 'Tweet' if 'Tweet' in df.columns else df.columns[0]
            label_col = 'Label' if 'Label' in df.columns else df.columns[1]

            df['clean_label'] = df[label_col].astype(str).str.strip().map(MAHAEMOTIONS_MAPPING)
            mapped = df.dropna(subset=['clean_label']).copy()
            mapped['target'] = mapped['clean_label'].map(LABEL_TO_ID)
            mapped['text'] = mapped[tweet_col].astype(str)
            results[name] = mapped[['text', 'clean_label', 'target']]
            print(f"Marathi MahaEmotions {name}: {len(mapped)} mapped samples across {mapped['clean_label'].nunique()} classes.")

    return results


def train_and_evaluate():
    """Execute complete training, calibration, and benchmarking workflow."""
    train_df, val_df, test_df, audit_stats = load_datasets()
    marathi_benchmarks = load_marathi_benchmark()

    # 1. Fit TF-IDF Vectorizer
    print("\nFitting TF-IDF Vectorizer (unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        tokenizer=tokenize,
        ngram_range=(1, 2),
        max_features=12000,
        sublinear_tf=True,
        token_pattern=None
    )

    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label'].values

    X_val = vectorizer.transform(val_df['text'])
    y_val = val_df['label'].values

    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label'].values

    print(f"TF-IDF Matrix Shape: {X_train.shape} (Vocabulary: {len(vectorizer.vocabulary_)} features)")

    # 2. Define Models
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42, solver='lbfgs'),
        "Linear SVC": LinearSVC(C=1.0, max_iter=2000, random_state=42)
    }

    trained_models = {}
    calibrated_models = {}
    evaluation_results = {}
    confusion_matrices = {}

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train, y_train)
        trained_models[name] = clf

        # For LinearSVC, create a calibrated probability wrapper via Platt scaling
        if isinstance(clf, LinearSVC):
            print("Calibrating Linear SVC decision boundaries for true probability output (3-fold CV)...")
            calibrated = CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, max_iter=2000, random_state=42),
                cv=3
            )
            calibrated.fit(X_train, y_train)
            calibrated_models[name] = calibrated
            pred_clf = calibrated
        else:
            calibrated_models[name] = clf
            pred_clf = clf

        # Predictions on Validation
        val_preds = pred_clf.predict(X_val)
        val_metrics = compute_metrics(y_val, val_preds)

        # Predictions on Test
        test_preds = pred_clf.predict(X_test)
        test_metrics = compute_metrics(y_test, test_preds)

        print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Test Accuracy:       {test_metrics['accuracy']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f}")

        evaluation_results[name] = {
            "validation": val_metrics,
            "test": test_metrics
        }
        confusion_matrices[name] = {
            "counts": test_metrics["confusion_matrix"],
            "normalized": test_metrics["confusion_matrix_normalized"],
            "labels": test_metrics["labels"]
        }

    # 3. Evaluate Cross-Lingual Transfer on Marathi MahaEmotions
    multilingual_eval = {}
    if "test" in marathi_benchmarks:
        marathi_test = marathi_benchmarks["test"]
        X_mr_test = vectorizer.transform(marathi_test['text'])
        y_mr_test = marathi_test['target'].values
        
        print("\n--- Evaluating Zero-Shot Cross-Lingual Transfer on Marathi Benchmark ---")
        for name in models.keys():
            clf = calibrated_models[name]
            mr_preds = clf.predict(X_mr_test)
            mr_metrics = compute_metrics(y_mr_test, mr_preds)
            print(f"  {name} on Marathi MahaEmotions Test -> Accuracy: {mr_metrics['accuracy']:.4f} | Macro F1: {mr_metrics['macro_f1']:.4f}")
            multilingual_eval[name] = {
                "benchmark": "MahaEmotions Marathi Test",
                "sample_count": len(marathi_test),
                "accuracy": mr_metrics["accuracy"],
                "macro_f1": mr_metrics["macro_f1"],
                "weighted_f1": mr_metrics["weighted_f1"],
                "per_class": mr_metrics["per_class"],
                "note": "Zero-shot evaluation with English TF-IDF vocabulary on native Devanagari script demonstrates genuine cross-lingual vocabulary divergence without translation mediation."
            }

    # 4. Save Artifacts
    print("\nSaving trained models, vectorizer, and evaluation metrics...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(trained_models["Multinomial Naive Bayes"], MULTINOMIAL_NB_PATH)
    joblib.dump(trained_models["Logistic Regression"], LOGISTIC_REGRESSION_PATH)
    # Save the calibrated Linear SVC for runtime inference & raw LinearSVC for feature weights
    joblib.dump(calibrated_models["Linear SVC"], LINEAR_SVC_PATH)
    joblib.dump(trained_models["Linear SVC"], MODELS_DIR / "raw_linear_svc.joblib")

    # Save metadata
    metadata = {
        "dataset_audit": audit_stats,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "emotion_classes": EMOTION_LABELS,
        "model_files": {
            "vectorizer": str(VECTORIZER_PATH.name),
            "multinomial_nb": str(MULTINOMIAL_NB_PATH.name),
            "logistic_regression": str(LOGISTIC_REGRESSION_PATH.name),
            "linear_svc": str(LINEAR_SVC_PATH.name)
        }
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)

    with open(CONFUSION_MATRICES_PATH, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices, f, indent=2)

    with open(MULTILINGUAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(multilingual_eval, f, indent=2)

    print("\nTraining and evaluation workflow successfully completed!")
    return evaluation_results


if __name__ == "__main__":
    train_and_evaluate()
