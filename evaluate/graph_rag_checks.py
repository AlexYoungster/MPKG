"""Integration checks for source-scoped Graph RAG and its HTTP endpoint."""

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from build_graph import ingest, load_source
from graph_rag import (GraphRAG, GraphRetriever, build_messages, prose_answer,
                        quantity_type_warning, _drop_mismatched_catalog_citations)
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


class AgentGenerator(FakeGenerator):
    def __init__(self, decisions):
        super().__init__()
        self.decisions = iter(decisions)
        self.agent_calls = []

    def agent_step(self, question, observation, relation_names, tools):
        self.agent_calls.append((question, observation, relation_names, tools))
        return next(self.decisions)


class GraphRAGChecks(unittest.TestCase):
    def test_model_agent_can_finish_from_complete_seed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            generator = AgentGenerator([
                {"action": "finish", "sufficient": True, "answer_mode": "single"},
            ])
            service = GraphRAG(service.retriever, generator, max_agent_steps=2)
            result = service.ask("What spindle speed was used for Part A?")
            self.assertEqual(result["retrieval"]["agent_status"], "sufficient")
            self.assertEqual(result["retrieval"]["agent_rounds"], 1)
            self.assertEqual(result["retrieval"]["agent_trace"], [])
            self.assertEqual(result["evidence"]["facts"][1]["object"], "1200 rpm")

    def test_model_agent_can_query_then_finish_multi_document_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            generator = AgentGenerator([
                {"action": "query", "tool": "filter_facts",
                 "arguments": {"relation": "Spindle Speed", "limit": 2}},
                {"action": "finish", "sufficient": True, "answer_mode": "list"},
            ])
            service = GraphRAG(service.retriever, generator, max_agent_steps=3)
            result = service.ask("列出所有工件的主轴转速")
            self.assertEqual(result["retrieval"]["agent_status"], "sufficient")
            self.assertEqual(result["retrieval"]["agent_rounds"], 2)
            self.assertEqual(result["retrieval"]["agent_trace"][0]["tool"], "filter_facts")
            self.assertEqual(result["retrieval"]["candidate_documents"], 2)
            self.assertIn("900 rpm", json.dumps(result))
            self.assertEqual(len(generator.agent_calls), 2)

    def test_model_agent_can_follow_a_two_hop_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            generator = AgentGenerator([
                {"action": "query", "tool": "find_paths",
                 "arguments": {"start_entity": "Part A",
                                "relation_path": ["Operation", "Spindle Speed"]}},
                {"action": "finish", "sufficient": True, "answer_mode": "single"},
            ])
            service = GraphRAG(service.retriever, generator, max_agent_steps=3)
            result = service.ask("Part A 的工序对应的主轴转速是多少？")
            self.assertEqual(result["retrieval"]["agent_status"], "sufficient")
            self.assertEqual(result["evidence"]["facts"][1]["object"], "1200 rpm")
            self.assertEqual(result["retrieval"]["agent_trace"][0]["tool"], "find_paths")

    def test_model_agent_is_bounded_when_it_does_not_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            generator = AgentGenerator([
                {"action": "query", "tool": "search_text",
                 "arguments": {"text_terms": ["turning"]}},
                {"action": "query", "tool": "search_text",
                 "arguments": {"text_terms": ["turning"]}},
            ])
            service = GraphRAG(service.retriever, generator, max_agent_steps=2)
            result = service.ask("请查找车削记录")
            self.assertEqual(result["retrieval"]["agent_status"], "step_limit")
            self.assertEqual(result["retrieval"]["agent_rounds"], 2)
            self.assertEqual(len(result["retrieval"]["agent_trace"]), 2)
            self.assertIn("重复查询", result["retrieval"]["agent_trace"][1]["error"])

    def test_agent_filter_tool_supports_bounded_pagination(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            first = service.retriever.execute_tool(
                "filter_facts", {"relation": "Spindle Speed", "limit": 1})
            second = service.retriever.execute_tool(
                "filter_facts", {"relation": "Spindle Speed", "limit": 1, "offset": 1})
            self.assertTrue(first["truncated"])
            self.assertEqual(first["candidate_documents"], 2)
            self.assertEqual(second["documents"][0]["source_line"], 2)

    def test_graph_query_accepts_model_match_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            result = service.retriever.execute_tool(
                "graph_query", {"query": {"kind": "match/documents",
                                             "relations": ["Spindle Speed"]}})
            self.assertEqual(result["candidate_documents"], 2)
            self.assertEqual(len(result["documents"]), 2)

    def test_graph_query_agent_can_create_a_catalogue_without_document_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            generator = AgentGenerator([
                {"action": "query", "tool": "graph_query",
                 "arguments": {"query": {"kind": "aggregate",
                                            "relation": "Operation",
                                            "limit": 10}}},
                {"action": "finish", "sufficient": True, "answer_mode": "list"},
            ])
            generator.generate = lambda messages: "工序包括 turning。[C1]"
            service = GraphRAG(service.retriever, generator, max_agent_steps=3)
            result = service.ask("现有记录中的工序有哪些？")
            self.assertEqual(result["retrieval"]["agent_status"], "sufficient")
            self.assertEqual(result["evidence"]["documents"], [])
            self.assertEqual([item["value"] for item in result["evidence"]["catalogs"]],
                             ["turning"])
            self.assertIn("Spindle Speed",
                          result["evidence"]["catalogs"][0]["outgoing_relations"])
            self.assertIn("[C1]", result["evidence_explanation"])
            self.assertEqual(result["retrieval"]["agent_trace"][0]["tool"], "graph_query")
            self.assertEqual(generator.agent_calls[0][3], ("graph_query",))

    def test_agent_owns_sufficiency_decision_for_truncated_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.make_service(Path(temporary))
            generator = AgentGenerator([
                {"action": "query", "tool": "filter_facts",
                 "arguments": {"relation": "Spindle Speed", "limit": 1}},
                {"action": "finish", "sufficient": True, "answer_mode": "list"},
            ])
            service = GraphRAG(service.retriever, generator, max_agent_steps=3)
            result = service.ask("列出所有工件的主轴转速")
            # The backend exposes truncation to the model but does not replace
            # the model's explicit verification decision with a hard-coded one.
            self.assertEqual(result["retrieval"]["agent_status"], "sufficient")
            self.assertTrue(result["retrieval"]["agent_assessment"]["sufficient"])
            self.assertTrue(result["retrieval"]["evidence_truncated"])

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

    def test_catalogue_citation_must_match_value_in_its_sentence(self):
        answer, citations = _drop_mismatched_catalog_citations(
            "精密加工包括 precision grinding [C1]，另有 precision turning [C2]。",
            ["C1", "C2"],
            [{"id": "C1", "value": "precision grinding"},
             {"id": "C2", "value": "hard turning"}],
        )
        self.assertIn("precision grinding [C1]", answer)
        self.assertNotIn("[C2]", answer)
        self.assertEqual(citations, ["C1"])

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
