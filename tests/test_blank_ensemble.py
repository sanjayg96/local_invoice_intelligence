import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "docile" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from baseline_plus import (  # noqa: E402
    DEFAULT_FIELD_SPECS,
    blank_ensemble_replacement_decision,
    field_record,
    should_attempt_blank_ensemble,
)


class BlankEnsembleTests(unittest.TestCase):
    def test_blank_ensemble_only_attempts_blank_fields(self):
        spec = DEFAULT_FIELD_SPECS["amount_total_gross"]

        self.assertTrue(should_attempt_blank_ensemble("blank", spec))
        self.assertFalse(should_attempt_blank_ensemble(None, spec))
        self.assertFalse(should_attempt_blank_ensemble("invalid_amount", spec))
        self.assertFalse(should_attempt_blank_ensemble("not_grounded", spec))

    def test_blank_ensemble_accepts_valid_grounded_candidate(self):
        spec = DEFAULT_FIELD_SPECS["amount_total_gross"]
        transcription = "Invoice total\nTotal Due $123.45"
        windows = [{"text": transcription, "score": 120.0}]
        primary = field_record(None, spec, transcription, "baseline_primary", "qwen3:14b", 1.0)
        candidate = field_record(
            "$123.45",
            spec,
            transcription,
            "baseline_rescue",
            "llama3.2:3b",
            0.5,
            evidence_quote="Total Due $123.45",
        )

        decision = blank_ensemble_replacement_decision(primary, candidate, spec, windows)

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "blank_primary_filled_by_valid_ensemble_candidate")

    def test_blank_ensemble_rejects_candidate_when_primary_is_not_blank(self):
        spec = DEFAULT_FIELD_SPECS["amount_total_gross"]
        transcription = "Invoice total\nTotal Due $123.45"
        windows = [{"text": transcription, "score": 120.0}]
        primary = field_record("$100.00", spec, transcription, "baseline_primary", "qwen3:14b", 1.0)
        candidate = field_record(
            "$123.45",
            spec,
            transcription,
            "baseline_rescue",
            "llama3.2:3b",
            0.5,
            evidence_quote="Total Due $123.45",
        )

        decision = blank_ensemble_replacement_decision(primary, candidate, spec, windows)

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "primary_not_blank")


if __name__ == "__main__":
    unittest.main()
