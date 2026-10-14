# ManSanket (मनसंकेत)
## Multilingual NLP-Based Emotion Detection System

> **Topic 10:** Emotion Detection from Text  
> **Tagline:** Understanding emotions through words.  
> **System Name:** ManSanket (मनसंकेत)  

---

## 1. Project Overview & Motivation

**ManSanket (मनसंकेत)** is an academic Natural Language Processing (NLP) and Machine Learning system engineered to detect fine-grained human emotional expressions from unstructured text across **English, Hindi, and Marathi**.

### Key Distinction: Emotion Detection vs. Sentiment Analysis
- **Sentiment Analysis:** Maps text onto a coarse scalar polarity spectrum (Positive, Negative, or Neutral).
- **Emotion Detection:** Dissects nuanced affective states. For example, both *Anger* and *Fear* fall under negative sentiment polarity, yet they stem from contrasting psychological appraisals (threat confrontation vs. threat avoidance) and prompt diametrically opposite behavioral responses.

---

## 2. Emotion Taxonomy

ManSanket implements the standardized **6-Class CARER Emotion Taxonomy**:

| Label ID | Emotion Category | Psychological Characterization | Representative Indicators |
|---|---|---|---|
| `0` | **Sadness** | Melancholy, grief, discouragement | *broken, exhausted, devastated, helpless* |
| `1` | **Joy** | Elation, happiness, contentment | *delighted, thrilled, wonderful, happy* |
| `2` | **Love** | Romantic affection, warmth, gratitude | *romantic, cherish, adore, affectionate* |
| `3` | **Anger** | Frustration, rage, resentment | *furious, disgusted, irritated, bitter* |
| `4` | **Fear** | Anxiety, panic, vulnerability | *terrified, anxious, nervous, scared* |
| `5` | **Surprise** | Astonishment, unexpected wonder | *stunned, astonished, amazed, shocked* |

---

## 3. System Architecture & NLP Pipeline

```
Raw Input Text
      │
      ▼
Unicode NFKC Normalization (Canonical glyph unification for Latin & Devanagari)
      │
      ▼
Sanitization & Contraction Expansion (Strict preservation of negation words)
      │
      ▼
Language & Script Identification (Devanagari: Hindi vs Marathi; Latin: English vs Hinglish vs Romanized Marathi)
      │
      ▼
Sublinear TF-IDF Feature Extraction (Unigrams + Bigrams, 12,000 features)
      │
      ▼
Classifier Evaluation & Calibration (Platt Sigmoid Calibration for true posterior probabilities)
      │
      ▼
Linear Feature Attribution & Explainability (Token contributions: W[c, i] × x[i])
      │
      ▼
Editorial UI Presentation (Localized in English, Hindi, and Marathi)
```

---

## 4. Benchmark Evaluation Results

All algorithms were trained on **15,969 deduplicated training samples** and evaluated against **2,000 held-out test samples** (English CARER benchmark) without data leakage.

### English Test Benchmark (2,000 Samples)

| Algorithm | Validation Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| **Linear SVC (Calibrated)** | **90.20%** | **88.25%** | **0.8407** | **0.8166** | **0.8266** | **0.8808** |
| **Logistic Regression** | 86.50% | 85.90% | 0.8143 | 0.7681 | 0.7864 | 0.8527 |
| **Multinomial Naive Bayes** | 71.95% | 73.40% | 0.7588 | 0.4908 | 0.5393 | 0.6974 |

### Marathi MahaEmotions Test Benchmark (774 Mapped Samples)

To evaluate zero-shot cross-lingual transfer without false claims, models were tested directly on native Devanagari text:

| Algorithm | Accuracy | Macro F1 | Linguistic Rationale |
|---|---|---|---|
| Linear SVC | 30.10% | 0.0926 | Zero-shot baseline on native Devanagari text; English TF-IDF vocabulary exhibits zero token overlap with Devanagari script (out-of-vocabulary), causing the model to predict the training majority class. |
| Translation-Mediated Route | ~65–75% | ~0.60–0.70 | Non-English text translated to English before evaluation against the validated model (requires explicit user consent). |

---

## 5. Explainable AI: Mathematical Feature Attribution

For linear classifiers (Linear SVC, Logistic Regression), the prediction score for class $c$ is:

$$\text{score}(c) = \sum_{i \in \text{tokens}} (x_i \cdot W_{c, i}) + b_c$$

Where:
- $x_i$ is the sublinear TF-IDF weight of token $i$: $x_i = (1 + \log(\text{tf}_i)) \cdot \text{idf}_i$.
- $W_{c, i}$ is the model weight learned for emotion class $c$ and token $i$.
- $x_i \cdot W_{c, i}$ is the **exact local feature attribution**. Positive scores increase the decision boundary toward the emotion; negative scores penalize it.

*Academic Disclaimer:* Feature attributions illustrate how the statistical classifier weighs vocabulary n-grams; they do not represent clinical or psychological assessments of human emotion.

---

## 6. Installation & Execution

### Prerequisites
- Python 3.10+ (macOS, Linux, or Windows)
- pip / virtualenv

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models & Generate Benchmarks
```bash
python -m src.train
```

### 3. Run Automated Test Suite
```bash
python -m pytest tests/
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Viva Preparation & Oral Examination Guide

### Q1: Why is Macro F1 reported rather than relying solely on Accuracy?
**Answer:** The dataset contains class imbalance (Joy accounts for ~33.5% and Sadness ~29.2%, while Surprise accounts for only ~3.6%). Overall accuracy is heavily dominated by majority classes. Macro F1 computes the unweighted arithmetic mean across all classes:
$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$
This penalizes models that ignore minority classes like Surprise or Love.

### Q2: Why did Linear SVC outperform Multinomial Naive Bayes?
**Answer:** Naive Bayes assumes that all feature words are conditionally independent given the class:
$$P(x_1, x_2, \dots, x_n \mid c) = \prod_{i=1}^n P(x_i \mid c)$$
In human language, words strongly co-occur (e.g., "very happy", "not good"). Linear SVC makes no independence assumptions and constructs an optimal maximum-margin hyperplane in the high-dimensional TF-IDF space.

### Q3: How are probabilities obtained from Linear SVC?
**Answer:** Standard Support Vector Machines output uncalibrated signed distances to the hyperplane $w^T x + b$. In ManSanket, we apply **Platt Scaling** via `CalibratedClassifierCV` (3-fold cross validation), which fits a logistic sigmoid function over the decision values to output true calibrated posterior probabilities $P(y = c \mid x)$.

### Q4: How does the system handle Devanagari script disambiguation?
**Answer:** Both Hindi and Marathi utilize the Devanagari script. ManSanket identifies:
1. Marathi-exclusive phonemes such as the letter **ळ** (`U+0933`).
2. Distinctive grammatical markers (Marathi: *आहे, नाही, मला, खूप, त्रास* vs. Hindi: *है, नहीं, मुझे, बहुत, दर्द*).

### Q5: What are the primary limitations of the current system?
**Answer:**
1. **Sarcasm and Irony:** Literal lexical models struggle when affective meaning contradicts surface words.
2. **Cross-Lingual Vocabulary Divergence:** Without multilingual dense transformers (such as MuRIL or XLM-RoBERTa), Devanagari words produce zero overlap with English TF-IDF matrices, necessitating translation-mediated inference.
