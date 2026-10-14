"""Language and Script Identification Module for ManSanket (मनसंकेत).

Distinguishes:
1. Scripts: Devanagari, Latin, Mixed, or Other.
2. Devanagari disambiguation: Marathi (mr) vs Hindi (hi) based on grammatical markers and phonemes (such as ळ).
3. Latin disambiguation: English (en) vs Hinglish vs Romanized Marathi (mr-latin).
4. Code-mixed text (Hinglish, Marathi-English).
5. Confidence scoring and graceful handling of short/ambiguous/unsupported inputs.
"""
import re
from typing import Dict, Any, List, Optional

try:
    from langdetect import detect_langs
except ImportError:
    detect_langs = None

# Unicode ranges
DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]')
LATIN_REGEX = re.compile(r'[a-zA-Z]')
TOKEN_REGEX = re.compile(r'[\w\u0900-\u097F]+', re.UNICODE)

# Distinctive Marathi Devanagari markers
MARATHI_DEVANAGARI_MARKERS = {
    'आहे', 'आहेत', 'नाही', 'नाहीत', 'मला', 'माझे', 'माझी', 'माझा', 'माझं',
    'होते', 'होता', 'होती', 'वाटत', 'वाटते', 'वाटतं', 'करणार', 'जाणार',
    'खूप', 'कसे', 'कसा', 'कशी', 'कसं', 'जीव', 'त्रास', 'शांतता', 'कंटाळा',
    'संपल', 'संपलं', 'झाले', 'झाला', 'झाली', 'असतं', 'केलं', 'त्याला', 'तिला',
    'आमचे', 'आमचा', 'कधी', 'नको', 'सांगू', 'विचार', 'काही', 'सगळं', 'आयुष्य',
    'आनंद', 'दुःख', 'राग', 'भीती', 'आश्चर्य', 'प्रेम', 'पाहिजे', 'येत', 'द्या',
    'आता', 'डोळे', 'काळजी', 'गेलो', 'गेली', 'स्वतः', 'मित्र', 'कुणी', 'कुणीही',
    'करायचं', 'जगावं', 'कळत', 'वाटतय'
}

# Distinctive Hindi Devanagari markers
HINDI_DEVANAGARI_MARKERS = {
    'है', 'हैं', 'नहीं', 'मुझे', 'मेरा', 'मेरी', 'मेरे', 'था', 'थी', 'थे',
    'लगता', 'लगती', 'लगते', 'करना', 'बहुत', 'क्यों', 'कैसे', 'कैसा',
    'कैसी', 'तकलीफ', 'जीना', 'दर्द', 'हो', 'रहा', 'रही', 'रहे',
    'हूँ', 'सकता', 'सकती', 'सकते', 'चाहता', 'चाहती', 'कुछ', 'अपने', 'अपनी',
    'अपना', 'कोई', 'क्या', 'अब', 'सब', 'जान', 'ज़िंदगी', 'जिंदगी', 'बात',
    'अकेला', 'परेशान', 'दोस्त', 'मन', 'हिम्मत', 'ख़त्म', 'खत्म', 'समझ',
    'खुशी', 'दुख', 'गुस्सा', 'डर', 'हैरानी', 'प्यार'
}

# Hinglish vocabulary markers
HINGLISH_MARKERS = {
    'mujhe', 'kuch', 'nahi', 'nahin', 'bohot', 'bahut', 'raha', 'rahi', 'rahe',
    'karna', 'karne', 'zindagi', 'dil', 'dard', 'akela', 'akeli', 'pareshan',
    'samajh', 'kyun', 'kyu', 'kaise', 'hoga', 'hogi', 'mera', 'meri', 'mere',
    'tum', 'aap', 'hum', 'thak', 'yaar', 'bhai', 'bhi', 'ab', 'sab', 'khatam',
    'apne', 'apna', 'apni', 'koi', 'kisi', 'karta', 'karti', 'lagta', 'lagti',
    'chahiye', 'khushi', 'gussa', 'pyaar', 'pyar', 'darr'
}

# Romanized Marathi vocabulary markers
MARATHI_LATIN_MARKERS = {
    'mala', 'kahi', 'nahi', 'khup', 'tras', 'hoto', 'hotay', 'aahe', 'ahe',
    'vatta', 'vatate', 'vatat', 'jeevan', 'ayushya', 'sagla', 'sampla', 'ata',
    'jagaycha', 'jagava', 'kantala', 'zalay', 'zhala', 'kasa', 'kashe',
    'mahit', 'vichar', 'sangu', 'shantata', 'sodun', 'gela', 'geli', 'amhi',
    'tyala', 'tila', 'nako', 'anand', 'raag', 'rag', 'bhiti', 'prem'
}


def analyze_script(text: str) -> Dict[str, Any]:
    """Calculate character ratios for Devanagari and Latin scripts."""
    total_chars = len(text)
    if total_chars == 0:
        return {"devanagari_ratio": 0.0, "latin_ratio": 0.0, "primary_script": "unknown"}

    devanagari_count = len(DEVANAGARI_REGEX.findall(text))
    latin_count = len(LATIN_REGEX.findall(text))
    
    alpha_chars = devanagari_count + latin_count
    if alpha_chars == 0:
        return {"devanagari_ratio": 0.0, "latin_ratio": 0.0, "primary_script": "symbol_or_numeric"}

    dev_ratio = devanagari_count / alpha_chars
    lat_ratio = latin_count / alpha_chars

    if dev_ratio >= 0.85:
        primary = "devanagari"
    elif lat_ratio >= 0.85:
        primary = "latin"
    elif dev_ratio >= 0.15 and lat_ratio >= 0.15:
        primary = "mixed"
    elif dev_ratio > lat_ratio:
        primary = "devanagari"
    else:
        primary = "latin"

    return {
        "devanagari_ratio": round(dev_ratio, 3),
        "latin_ratio": round(lat_ratio, 3),
        "primary_script": primary
    }


def detect_language(text: str) -> Dict[str, Any]:
    """Identify the language and code-mixing status of user-provided text.
    
    Returns:
        Dict containing:
        - language_code: 'en', 'hi', 'mr', 'hinglish', 'mr-en-codemixed', 'unsupported', 'unknown'
        - language_name: Human readable name
        - script: 'devanagari', 'latin', 'mixed', 'unknown'
        - confidence: Float 0.0 to 1.0
        - is_short_or_ambiguous: Boolean
        - is_supported: Boolean
        - details: Descriptive explanation of detection rationale
    """
    if not text or not text.strip():
        return {
            "language_code": "unknown",
            "language_name": "Empty Input",
            "script": "unknown",
            "confidence": 0.0,
            "is_short_or_ambiguous": True,
            "is_supported": False,
            "details": "No text provided."
        }

    stripped = text.strip()
    words = TOKEN_REGEX.findall(stripped.lower())
    total_words = len(words)
    total_chars = len(stripped)

    # Short input check
    is_short = total_words <= 2 or total_chars < 10

    script_info = analyze_script(stripped)
    primary_script = script_info["primary_script"]

    # 1. Devanagari Script Analysis
    if primary_script == "devanagari":
        # Check for Marathi-specific letter ळ (U+0933)
        has_marathi_lla = 'ळ' in stripped
        
        # Word set matches
        marathi_hits = sum(1 for w in words if w in MARATHI_DEVANAGARI_MARKERS)
        hindi_hits = sum(1 for w in words if w in HINDI_DEVANAGARI_MARKERS)

        if has_marathi_lla or marathi_hits > hindi_hits:
            confidence = 0.90 if not is_short else 0.65
            return {
                "language_code": "mr",
                "language_name": "Marathi (Devanagari)",
                "script": "devanagari",
                "confidence": confidence,
                "is_short_or_ambiguous": is_short,
                "is_supported": True,
                "details": f"Devanagari script with Marathi markers ({marathi_hits} markers, ळ present: {has_marathi_lla})."
            }
        elif hindi_hits > marathi_hits:
            confidence = 0.90 if not is_short else 0.65
            return {
                "language_code": "hi",
                "language_name": "Hindi (Devanagari)",
                "script": "devanagari",
                "confidence": confidence,
                "is_short_or_ambiguous": is_short,
                "is_supported": True,
                "details": f"Devanagari script with Hindi markers ({hindi_hits} markers)."
            }
        else:
            # Ambiguous Devanagari text without strong lexical markers
            return {
                "language_code": "hi",  # default Devanagari candidate
                "language_name": "Hindi / Marathi (Devanagari)",
                "script": "devanagari",
                "confidence": 0.50,
                "is_short_or_ambiguous": True,
                "is_supported": True,
                "details": "Devanagari script detected, but lexical markers are insufficient to distinguish Hindi from Marathi. Please verify manually."
            }

    # 2. Mixed Script Analysis (Devanagari + Latin)
    if primary_script == "mixed":
        return {
            "language_code": "code-mixed",
            "language_name": "Code-Mixed (Devanagari + Latin)",
            "script": "mixed",
            "confidence": 0.75,
            "is_short_or_ambiguous": is_short,
            "is_supported": True,
            "details": f"Mixed script text (Devanagari: {int(script_info['devanagari_ratio']*100)}%, Latin: {int(script_info['latin_ratio']*100)}%)."
        }

    # 3. Latin Script Analysis
    if primary_script == "latin":
        hinglish_hits = sum(1 for w in words if w in HINGLISH_MARKERS)
        marathi_latin_hits = sum(1 for w in words if w in MARATHI_LATIN_MARKERS)

        if hinglish_hits > 0 and hinglish_hits >= marathi_latin_hits:
            return {
                "language_code": "hinglish",
                "language_name": "Hinglish (Romanized Hindi)",
                "script": "latin",
                "confidence": 0.85 if hinglish_hits >= 2 else 0.60,
                "is_short_or_ambiguous": is_short,
                "is_supported": True,
                "details": f"Latin alphabet with {hinglish_hits} Hinglish markers."
            }
        elif marathi_latin_hits > 0:
            return {
                "language_code": "mr-en-codemixed",
                "language_name": "Marathi-English (Romanized Marathi)",
                "script": "latin",
                "confidence": 0.85 if marathi_latin_hits >= 2 else 0.60,
                "is_short_or_ambiguous": is_short,
                "is_supported": True,
                "details": f"Latin alphabet with {marathi_latin_hits} Romanized Marathi markers."
            }

        # Fallback to langdetect if available
        if detect_langs:
            try:
                candidates = detect_langs(stripped)
                if candidates:
                    top_cand = candidates[0]
                    if top_cand.lang == "en":
                        return {
                            "language_code": "en",
                            "language_name": "English",
                            "script": "latin",
                            "confidence": round(float(top_cand.prob), 2),
                            "is_short_or_ambiguous": is_short or (top_cand.prob < 0.75),
                            "is_supported": True,
                            "details": f"Statistical language model detected English (prob: {top_cand.prob:.2f})."
                        }
                    elif top_cand.lang in ["hi", "mr"]:
                        lang_name = "Hindi" if top_cand.lang == "hi" else "Marathi"
                        return {
                            "language_code": top_cand.lang,
                            "language_name": lang_name,
                            "script": "latin",
                            "confidence": round(float(top_cand.prob), 2),
                            "is_short_or_ambiguous": is_short,
                            "is_supported": True,
                            "details": f"Detected {lang_name} in Latin script."
                        }
                    else:
                        return {
                            "language_code": "unsupported",
                            "language_name": f"Unsupported ({top_cand.lang.upper()})",
                            "script": "latin",
                            "confidence": round(float(top_cand.prob), 2),
                            "is_short_or_ambiguous": is_short,
                            "is_supported": False,
                            "details": f"Detected language code '{top_cand.lang}', which is not currently supported by ManSanket."
                        }
            except Exception:
                pass

        # Default Latin assumption when detector is uncertain
        return {
            "language_code": "en",
            "language_name": "English",
            "script": "latin",
            "confidence": 0.65 if not is_short else 0.45,
            "is_short_or_ambiguous": is_short,
            "is_supported": True,
            "details": "Assumed English in Latin script; short or unclassified text."
        }

    return {
        "language_code": "unknown",
        "language_name": "Unidentified Script",
        "script": primary_script,
        "confidence": 0.20,
        "is_short_or_ambiguous": True,
        "is_supported": False,
        "details": "Non-alphabetic or unclassifiable script."
    }
