"""Checks that CoT decisions come from the final answer and preserve type checks."""

import unittest
import json
import tempfile
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from edc.edc_framework import EDC
from edc.schema_canonicalization_cot import SchemaCanonicalizerCoT


class CoTChecks(unittest.TestCase):
    def test_pipeline_writes_cot_trace(self):
        pipeline = EDC.__new__(EDC)
        for name in ("oie_llm_name", "sd_llm_name", "sc_llm_name",
                     "sc_embedder_name", "ee_llm_name", "sr_embedder_name"):
            setattr(pipeline, name, "test-model")
        pipeline.sc_cot = True
        source = ["零件", "材料", "钢"]
        pipeline.oie = lambda *args, **kwargs: ([[source]], [""], [""])
        pipeline.schema_definition = lambda *args, **kwargs: [{"材料": "构成材料"}]

        def canonicalize(self, *args, **kwargs):
            self.last_cot_trace_by_entry = [[{"query_triplet": source, "selected_option": "A"}]]
            return [[source]], [[{}]]

        pipeline.schema_canonicalization = MethodType(canonicalize, pipeline)
        with tempfile.TemporaryDirectory() as directory:
            pipeline.extract_kg(["零件由钢制成"], str(Path(directory) / "run"))
            result_path = Path(directory) / "run" / "iter0" / "result_at_each_stage.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result[0]["canonicalization_reasoning"][0]["selected_option"], "A")

    def test_final_answer_is_selected_over_reasoning_mentions(self):
        parse = SchemaCanonicalizerCoT.extract_final_option
        self.assertEqual(parse("候选 A 和 B 均需比较。\n最终答案：C", set("ABC")), "C")
        self.assertIsNone(parse("候选 A 和 B 均需比较。", set("ABC")))
        self.assertEqual(parse("分析完成\n选项 B", set("ABC")), "B")

    def test_unique_semantic_family_guard_is_preserved(self):
        verifier = SchemaCanonicalizerCoT.__new__(SchemaCanonicalizerCoT)
        verifier.schema_dict = {
            "切削速度": "刀具相对于工件的切削线速度。",
            "进给量": "刀具相对于工件的每转进给距离。",
        }
        verifier.verifier_model = object()
        verifier.verifier_tokenizer = object()
        verifier.verifier_openai_model = None
        verifier.max_tokens = 128
        verifier.verification_trace = []
        source = ["车削加工", "进给率", "0.1mm/r"]
        with patch(
            "edc.schema_canonicalization_cot.llm_utils.generate_completion_transformers",
            return_value="A 看似接近，但与进给参数不同。\n最终答案：A",
        ):
            actual = verifier.llm_verify(
                "车削加工进给率为0.1mm/r。", source,
                "表示每转进给距离", "{input_text}\n{choices}",
                {"切削速度": verifier.schema_dict["切削速度"],
                 "进给量": verifier.schema_dict["进给量"]},
            )
        self.assertEqual(actual, ["车削加工", "进给量", "0.1mm/r"])
        self.assertTrue(verifier.verification_trace[0]["family_override"])
        self.assertEqual(source, ["车削加工", "进给率", "0.1mm/r"])


if __name__ == "__main__":
    unittest.main()
