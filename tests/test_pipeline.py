"""Integration tests for the inference and explainability pipeline in ManSanket."""
import unittest
from src.predict import predict_emotion, load_inference_models
from src.explainability import explain_prediction, get_global_top_features
from src.i18n import t


class TestPipeline(unittest.TestCase):

    def test_model_loading(self):
        models = load_inference_models()
        self.assertIn("vectorizer", models)
        self.assertIn("Linear SVC", models)
        self.assertIn("Logistic Regression", models)
        self.assertIn("Multinomial Naive Bayes", models)

    def test_prediction_output_structure(self):
        sample = "I feel so genuinely happy, delighted, and energized by this wonderful achievement!"
        res = predict_emotion(sample, model_name="Linear SVC")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["predicted_label"], "joy")
        self.assertGreater(res["confidence"], 0.70)
        self.assertIn("probabilities", res)
        self.assertEqual(len(res["probabilities"]), 6)
        # Check sum of probabilities close to 1.0
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_explainability_attribution(self):
        sample = "I am feeling so terrified and scared of this dark room"
        res = predict_emotion(sample, model_name="Linear SVC")
        self.assertEqual(res["predicted_label"], "fear")
        self.assertTrue(len(res["top_features"]) > 0)
        
        # Check that 'terrified' or 'scared' appears in top features
        top_words = [f["feature"] for f in res["top_features"]]
        self.assertTrue(any(w in top_words for w in ["terrified", "scared", "fear"]))

    def test_global_features(self):
        global_feats = get_global_top_features(n_features=5, model_name="Linear SVC")
        self.assertIn("joy", global_feats)
        self.assertIn("sadness", global_feats)
        self.assertEqual(len(global_feats["joy"]), 5)

    def test_i18n_translation_and_fallback(self):
        # English
        self.assertEqual(t("nav.overview", "en"), "Overview")
        # Hindi
        self.assertIn("अवलोकन", t("nav.overview", "hi"))
        # Marathi
        self.assertIn("परिचय", t("nav.overview", "mr"))
        # Fallback for unknown key
        self.assertEqual(t("unknown.nonexistent.key", "en"), "key")


if __name__ == "__main__":
    unittest.main()
