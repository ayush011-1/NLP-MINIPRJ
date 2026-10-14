"""Unit tests for NLP preprocessing and normalization in ManSanket."""
import unittest
from src.preprocessing import clean_text, expand_contractions, tokenize, compute_text_statistics


class TestPreprocessing(unittest.TestCase):

    def test_empty_and_none(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(tokenize(""), [])

    def test_unicode_normalization(self):
        # Devanagari decomposed vs composed glyphs (NFKC)
        raw_decomposed = "क" + "\u093E"  # ka + aa matra
        cleaned = clean_text(raw_decomposed)
        self.assertTrue(len(cleaned) > 0)
        self.assertIn("का", cleaned)

    def test_url_and_handle_removal(self):
        sample = "Check out https://example.com/test and email me at test@example.com @user123 great!"
        cleaned = clean_text(sample)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("test@example.com", cleaned)
        self.assertNotIn("@user123", cleaned)
        self.assertIn("Check out", cleaned)
        self.assertIn("great!", cleaned)

    def test_contraction_expansion_preserves_negation(self):
        sample = "I didn't want this and won't accept it, isn't that clear?"
        expanded = expand_contractions(sample)
        self.assertIn("did not", expanded)
        self.assertIn("will not", expanded)
        self.assertIn("is not", expanded)
        self.assertNotIn("didn't", expanded)

    def test_tokenization_devanagari_and_latin(self):
        sample_mr = "मला खूप आनंद झाला आहे"
        tokens = tokenize(sample_mr)
        self.assertEqual(tokens, ["मला", "खूप", "आनंद", "झाला", "आहे"])

        sample_en = "I am so happy!"
        tokens_en = tokenize(sample_en)
        self.assertEqual(tokens_en, ["i", "am", "so", "happy"])

    def test_text_statistics(self):
        stats = compute_text_statistics("Hello world. How are you today?")
        self.assertEqual(stats["word_count"], 6)
        self.assertEqual(stats["sentence_count"], 2)
        self.assertGreater(stats["avg_word_length"], 0)
        self.assertGreater(stats["lexical_diversity"], 0)


if __name__ == "__main__":
    unittest.main()
