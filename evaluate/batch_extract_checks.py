"""Small integration checks for batching, failure isolation and resumption."""

import json
import tempfile
import unittest
from pathlib import Path

from batch_extract import main


class FakeEDC:
    calls = 0

    def __init__(self, **settings):
        self.settings = settings

    def extract_kg(self, texts, output_dir):
        FakeEDC.calls += 1
        if any("FAIL" in text for text in texts):
            raise RuntimeError("simulated failure")
        target = Path(output_dir) / "iter0"
        target.mkdir(parents=True)
        stages = [{
            "input_text": text,
            "oie": [["operation", "Tool", "carbide tool"]],
            "schema_definition": {"Tool": "uses a tool"},
            "schema_canonicalizaiton": [["operation", "Tool", "carbide tool"]],
        } for text in texts]
        (target / "result_at_each_stage.json").write_text(
            json.dumps(stages), encoding="utf-8"
        )


class BatchExtractTest(unittest.TestCase):
    def test_english_definition_fallback_uses_relation_meaning(self):
        from edc.schema_definition import SchemaDefiner

        example = [["face milling operation", "Cutting Speed", "180 m/min"]]
        definition = SchemaDefiner._generic_instance_definition("Cutting Speed", example)
        self.assertIn("cutting velocity parameter", definition)
        self.assertNotIn("surface roughness parameter", definition)
        depth = SchemaDefiner._generic_instance_definition(
            "Cutting Depth", [["face milling operation", "Cutting Depth", "2.5 mm"]]
        )
        self.assertIn("cutting depth parameter", depth)

    def test_failure_isolation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.txt"
            output = root / "output"
            dataset.write_text(
                "operation uses carbide tool\nFAIL\n\noperation uses carbide tool\n",
                encoding="utf-8",
            )
            argv = ["--input", str(dataset), "--output-dir", str(output), "--chunk-size", "3"]
            FakeEDC.calls = 0
            summary = main(argv, edc_factory=FakeEDC)
            self.assertEqual(summary["nonblank_lines"], 3)
            self.assertEqual(summary["checked_lines"], 2)
            self.assertEqual(summary["pipeline_error_lines"], 1)
            self.assertEqual(summary["structural_and_schema_compliant_lines"], 2)
            self.assertEqual(summary["canonical_triples"], 2)
            self.assertEqual(summary["schema_relation_membership_rate"], 1.0)
            self.assertEqual(len((output / "triples.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(len((output / "records.jsonl").read_text(encoding="utf-8").splitlines()), 4)
            first_calls = FakeEDC.calls
            resumed = main(argv + ["--resume"], edc_factory=FakeEDC)
            self.assertEqual(resumed["canonical_triples"], 2)
            self.assertEqual(FakeEDC.calls, first_calls + 1)


if __name__ == "__main__":
    unittest.main()
