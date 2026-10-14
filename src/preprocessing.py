"""Text preprocessing and normalization pipeline for ManSanket (मनसंकेत).

Handles:
- Unicode NFKC normalization for clean Devanagari and Latin processing
- URL, email, and user handle sanitization
- Contraction expansion with strict preservation of emotional negations
- Script-aware tokenization
- Linguistic summary statistics
"""
import re
import unicodedata
from typing import List, Dict, Any, Optional

# Regex patterns for cleaning
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
HANDLE_PATTERN = re.compile(r'@\w+')
MULTIPLE_SPACES_PATTERN = re.compile(r'\s+')
REPEATED_PUNCTUATION_PATTERN = re.compile(r'([!?.।॥]){2,}')

# English Contractions Dictionary (crucial for preserving negations)
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that's": "that is",
    "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is",
    "who'll": "who will", "who's": "who is", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

# Unicode-aware token pattern: extracts letters, digits, and combining marks across Latin and Devanagari
TOKEN_PATTERN = re.compile(r'[\w\u0900-\u097F]+', re.UNICODE)


def expand_contractions(text: str) -> str:
    """Expand common English contractions while preserving lower/upper casing."""
    for contraction, expansion in CONTRACTIONS.items():
        # Match case-insensitively with word boundary
        pattern = re.compile(r'\b' + re.escape(contraction) + r'\b', re.IGNORECASE)
        text = pattern.sub(expansion, text)
    return text


def clean_text(text: Optional[str], expand_contr: bool = True) -> str:
    """Sanitize and normalize text while preserving Devanagari and emotional cues.
    
    Args:
        text: Raw input string.
        expand_contr: Whether to expand English contractions.
        
    Returns:
        Cleaned, normalized string.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unicode NFKC Normalization (vital for Devanagari glyphs and combining marks)
    cleaned = unicodedata.normalize("NFKC", text)

    # 2. Remove URLs, email addresses, and user handles
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = EMAIL_PATTERN.sub(" ", cleaned)
    cleaned = HANDLE_PATTERN.sub(" ", cleaned)

    # 3. Collapse excessive repeated punctuation (e.g., '!!!!' -> '!')
    cleaned = REPEATED_PUNCTUATION_PATTERN.sub(r'\1', cleaned)

    # 4. Expand contractions
    if expand_contr:
        cleaned = expand_contractions(cleaned)

    # 5. Normalize whitespace
    cleaned = MULTIPLE_SPACES_PATTERN.sub(" ", cleaned).strip()

    return cleaned


def tokenize(text: str, lower: bool = True) -> List[str]:
    """Extract tokens preserving Devanagari and Latin characters.
    
    Args:
        text: Input string.
        lower: Whether to lowercase Latin characters.
        
    Returns:
        List of string tokens.
    """
    if not text:
        return []
    
    if lower:
        text = text.lower()
        
    return TOKEN_PATTERN.findall(text)


def compute_text_statistics(text: str) -> Dict[str, Any]:
    """Calculate basic linguistic and structural statistics of the text.
    
    Args:
        text: Input string.
        
    Returns:
        Dictionary with character count, word count, token count, average word length, etc.
    """
    if not text or not isinstance(text, str):
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "unique_word_count": 0,
            "lexical_diversity": 0.0
        }

    raw_clean = clean_text(text, expand_contr=False)
    tokens = tokenize(raw_clean, lower=True)
    word_count = len(tokens)
    unique_words = set(tokens)
    unique_count = len(unique_words)
    
    # Sentence estimate based on common terminators (., ?, !, ।, ॥)
    sentences = [s.strip() for s in re.split(r'[.!?।॥\n]+', text) if s.strip()]
    sentence_count = max(1, len(sentences)) if text.strip() else 0
    
    avg_len = (sum(len(t) for t in tokens) / word_count) if word_count > 0 else 0.0
    diversity = (unique_count / word_count) if word_count > 0 else 0.0

    return {
        "char_count": len(text),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_len, 2),
        "unique_word_count": unique_count,
        "lexical_diversity": round(diversity, 3)
    }
