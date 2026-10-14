"""Unit tests for language and script identification in ManSanket."""
import unittest
from src.language_detection import detect_language, analyze_script


class TestLanguageDetection(unittest.TestCase):

    def test_empty_input(self):
        res = detect_language("")
        self.assertEqual(res["language_code"], "unknown")
        self.assertTrue(res["is_short_or_ambiguous"])

    def test_marathi_devanagari_detection(self):
        # Using distinctive Marathi word 'आहे' and 'मला'
        mr_text = "मला आज खूप आनंद झाला आहे आणि मी खूप समाधानी आहे"
        res = detect_language(mr_text)
        self.assertEqual(res["language_code"], "mr")
        self.assertEqual(res["script"], "devanagari")
        self.assertTrue(res["is_supported"])

    def test_marathi_lla_phoneme(self):
        # Letter ळ is uniquely Marathi among major Devanagari languages
        mr_text = "सगळा निकाल जाहीर झाला"
        res = detect_language(mr_text)
        self.assertEqual(res["language_code"], "mr")
        self.assertEqual(res["script"], "devanagari")

    def test_hindi_devanagari_detection(self):
        hi_text = "मुझे आज बहुत गुस्सा आ रहा है और मैं बहुत परेशान हूँ"
        res = detect_language(hi_text)
        self.assertEqual(res["language_code"], "hi")
        self.assertEqual(res["script"], "devanagari")
        self.assertTrue(res["is_supported"])

    def test_english_latin_detection(self):
        en_text = "I feel incredibly joyful and motivated to achieve my dreams today"
        res = detect_language(en_text)
        self.assertEqual(res["language_code"], "en")
        self.assertEqual(res["script"], "latin")
        self.assertTrue(res["is_supported"])

    def test_hinglish_detection(self):
        hinglish_text = "Mujhe bohot zyada darr lag raha hai exam ke results se yaar"
        res = detect_language(hinglish_text)
        self.assertEqual(res["language_code"], "hinglish")
        self.assertEqual(res["script"], "latin")

    def test_short_input_flag(self):
        short_text = "Hi"
        res = detect_language(short_text)
        self.assertTrue(res["is_short_or_ambiguous"])


if __name__ == "__main__":
    unittest.main()
