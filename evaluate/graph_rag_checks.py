"""Integration checks for source-scoped Graph RAG and its HTTP endpoint."""

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from build_graph import ingest, load_source
from graph_rag import GraphRAG, GraphRetriever, build_messages, prose_answer, quantity_type_warning
from qa_backend import handler_for


class FakeGenerator:
    model = None

    def __init__(self):
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return "该工件采用所示工序，主轴转速见图谱记录。[E2]"

    def translate_for_lookup(self, question):
        return "What spindle speed was used for Part A?" if "甲" in question else question


class GraphRAGChecks(unittest.TestCase):
    def test_quantity_is_not_used_as_category_name(self):
        self.assertTrue(quantity_type_warning("Coolant", "8 L/min"))
        self.assertFalse(quantity_type_warning("Spindle Speed", "1200 rpm"))
        self.assertFalse(quantity_type_warning("Coolant", "cutting fluid"))

    def test_raw_triple_listing_is_kept_out_of_answer_paragraph(self):
        answer, citations = prose_answer(
            "钢轴的主轴转速为1200 rpm。[E2] turning --Spindle Speed--> 1200 rpm。",
            {"E2"},
        )
        self.assertNotIn("--", answer)
        self.assertIn("1200 rpm", answer)
        self.assertIn("[E2]", answer)
        self.assertEqual(citations, ["E2"])

    def make_service(self, root):
        schema = root / "schema.csv"
        schema.write_text(
            "Operation,The part uses the operation.\n"
            "Spindle Speed,The operation has the given spindle speed.\n",
            encoding="utf-8",
        )
        stages = root / "result_at_each_stage.json"
        stages.write_text(json.dumps([
            {"index": 0, "input_text": "Part A uses turning at 1200 rpm.",
             "schema_canonicalizaiton": [["Part A", "Operation", "turning"],
                                          ["turning", "Spindle Speed", "1200 rpm"]]},
            {"index": 1, "input_text": "Part B uses turning at 900 rpm.",
             "schema_canonicalizaiton": [["Part B", "Operation", "turning"],
                                          ["turning", "Spindle Speed", "900 rpm"]]},
        ]), encoding="utf-8")
        database = root / "graph.sqlite"
        ingest(load_source(stages, False), schema, database)
        generator = FakeGenerator()
        return GraphRAG(GraphRetriever(database), generator), generator

    def test_workpiece_answer_stays_with_its_source_document(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, generator = self.make_service(Path(temporary))
            result = service.ask("What spindle speed was used for Part A?")
            self.assertEqual(result["retrieval"]["anchor"], "Part A")
            self.assertEqual(result["retrieval"]["candidate_documents"], 1)
            self.assertEqual({fact["object"] for fact in result["evidence"]["facts"]},
                             {"turning", "1200 rpm"})
            self.assertNotIn("900 rpm", json.dumps(result))
            self.assertEqual(result["cited_evidence"], ["E2"])
            self.assertIn("Part A uses turning at 1200 rpm.", result["evidence_explanation"])
            self.assertNotIn("第 1 行", result["evidence_explanation"])
            self.assertEqual(generator.calls, 1)

    def test_conflicting_category_fact_is_marked_for_source_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            retrieved = service.retriever.retrieve("Part A operation")
            retrieved["facts"][1]["relation"] = "Coolant"
            retrieved["facts"][1]["object"] = "8 L/min"
            retrieved["facts"][1]["type_warning"] = True
            prompt = build_messages("What coolant?", retrieved)[1]["content"]
            self.assertIn("类型不一致", prompt)
            self.assertNotIn("Coolant--> 8 L/min", prompt)

    def test_answer_reusing_conflicting_quantity_is_withheld(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.csv"
            schema.write_text("Coolant,The operation uses the cooling medium.\n",
                              encoding="utf-8")
            stage = root / "result_at_each_stage.json"
            stage.write_text(json.dumps([{
                "index": 0, "input_text": "Part A used cutting fluid at 8 L/min.",
                "schema_canonicalizaiton": [["Part A", "Coolant", "8 L/min"]],
            }]), encoding="utf-8")
            database = root / "graph.sqlite"
            ingest(load_source(stage, False), schema, database)

            class UnsafeGenerator:
                def generate(self, messages):
                    return "冷却液是8 L/min。[D1]"

            result = GraphRAG(GraphRetriever(database), UnsafeGenerator()).ask(
                "What coolant does Part A use?")
            self.assertEqual(result["grounding_warning"], "category_quantity_conflict")
            self.assertIn("无法通过证据校验", result["answer"])
            self.assertIn("[D1]", result["evidence_explanation"])
            self.assertIn("Part A used cutting fluid at 8 L/min.", result["evidence_explanation"])

    def test_missing_entity_abstains_without_model_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, generator = self.make_service(Path(temporary))
            result = service.ask("What is its speed?", entity="Part C")
            self.assertEqual(result["evidence"]["facts"], [])
            self.assertIn("没有找到", result["answer"])
            self.assertEqual(generator.calls, 0)

    def test_translated_question_uses_exact_entity(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, generator = self.make_service(Path(temporary))
            result = service.ask("甲件的主轴转速是多少？")
            self.assertEqual(result["retrieval"]["mode"], "translated_exact_entity")
            self.assertEqual(result["retrieval"]["anchor"], "Part A")
            self.assertEqual(result["evidence"]["facts"][1]["object"], "1200 rpm")
            self.assertEqual(generator.calls, 1)

    def test_shared_process_requires_workpiece_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, generator = self.make_service(Path(temporary))
            result = service.ask("What spindle speed was used for turning?")
            self.assertTrue(result["retrieval"]["ambiguous"])
            self.assertEqual(result["retrieval"]["candidate_documents"], 2)
            self.assertIn("无法确定", result["answer"])
            self.assertIn("Part A uses turning at 1200 rpm.", result["evidence_explanation"])
            self.assertIn("Part B uses turning at 900 rpm.", result["evidence_explanation"])
            self.assertNotIn("第 1 行", result["evidence_explanation"])
            self.assertEqual(generator.calls, 0)

    def test_http_ask_returns_evidence_and_health(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            server = HTTPServer(("127.0.0.1", 0), handler_for(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(base + "/health") as response:
                    self.assertEqual(json.load(response)["status"], "ok")
                request = urllib.request.Request(
                    base + "/ask",
                    data=json.dumps({"question": "Part B spindle speed?"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    result = json.load(response)
                self.assertEqual(result["evidence"]["facts"][1]["object"], "900 rpm")
                self.assertEqual(result["evidence"]["documents"][0]["source_line"], 2)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
