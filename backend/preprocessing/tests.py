from django.test import SimpleTestCase

from preprocessing.preprocessing_service import (
    TextPreprocessor,
)


class TextPreprocessorTests(SimpleTestCase):
    def test_quora_profile_uses_stemming(self):
        preprocessor = TextPreprocessor(
            dataset_key="quora"
        )

        tokens = preprocessor.preprocess_tokens(
            "What causes nightmares?"
        )

        self.assertIn("caus", tokens)
        self.assertIn("nightmar", tokens)

        self.assertTrue(
            preprocessor.use_stemming
        )

    def test_quora_preserves_negation(self):
        preprocessor = TextPreprocessor(
            dataset_key="quora"
        )

        tokens = preprocessor.preprocess_tokens(
            "Why does this not work?"
        )

        self.assertIn("not", tokens)

    def test_quora_expands_negative_contractions(self):
        preprocessor = TextPreprocessor(
            dataset_key="quora"
        )

        tokens = preprocessor.preprocess_tokens(
            "Why doesn't this treatment work?"
        )

        self.assertIn("not", tokens)

    def test_clinical_profile_preserves_medical_tokens(self):
        preprocessor = TextPreprocessor(
            dataset_key="clinical_trials"
        )

        tokens = preprocessor.preprocess_tokens(
            "BRAF V600E HER2-positive "
            "64-year-old 20 mg/kg/day"
        )

        required_tokens = {
            "braf",
            "v600e",
            "her2-positive",
            "64-year-old",
            "20",
            "mg/kg/day",
        }

        self.assertTrue(
            required_tokens.issubset(
                set(tokens)
            )
        )

    def test_clinical_profile_does_not_stem_words(self):
        preprocessor = TextPreprocessor(
            dataset_key="clinical_trials"
        )

        tokens = preprocessor.preprocess_tokens(
            "mutations treatments"
        )

        self.assertIn(
            "mutations",
            tokens,
        )

        self.assertIn(
            "treatments",
            tokens,
        )

        self.assertFalse(
            preprocessor.use_stemming
        )

    def test_clinical_profile_preserves_negation(self):
        preprocessor = TextPreprocessor(
            dataset_key="clinical_trials"
        )

        tokens = preprocessor.preprocess_tokens(
            "Patients without prior chemotherapy"
        )

        self.assertIn(
            "without",
            tokens,
        )

    def test_profiles_have_different_configurations(self):
        quora = TextPreprocessor(
            dataset_key="quora"
        )

        clinical = TextPreprocessor(
            dataset_key="clinical_trials"
        )

        self.assertNotEqual(
            quora.get_configuration(),
            clinical.get_configuration(),
        )

        self.assertTrue(
            quora.get_configuration()[
                "use_stemming"
            ]
        )

        self.assertFalse(
            clinical.get_configuration()[
                "use_stemming"
            ]
        )

    def test_unknown_dataset_uses_default_profile(self):
        preprocessor = TextPreprocessor(
            dataset_key="unknown_dataset"
        )

        configuration = (
            preprocessor.get_configuration()
        )

        self.assertEqual(
            configuration["dataset_key"],
            "unknown_dataset",
        )

        self.assertTrue(
            configuration["use_stemming"]
        )

    def test_invalid_minimum_token_length_is_rejected(self):
        with self.assertRaises(ValueError):
            TextPreprocessor(
                dataset_key="quora",
                minimum_token_length=0,
            )

    def test_non_english_language_is_rejected(self):
        with self.assertRaises(ValueError):
            TextPreprocessor(
                dataset_key="quora",
                language="arabic",
            )