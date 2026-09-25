"""Integration checks for graph ingestion, replacement, and provenance."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from build_graph import ingest, load_source, neighbors, stats


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


class GraphBuildChecks(unittest.TestCase):
    def test_batch_deduplicates_edges_and_preserves_every_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch"
            batch.mkdir()
            schema = root / "schema.csv"
            database = root / "graph.sqlite"
            schema.write_text("Tool,The process uses the tool.\n", encoding="utf-8")
            records = [
                {"index": index, "source_line": index + 1, "input_text": "milling uses cutter",
                 "status": "checked", "flags": [], "canonical_count": 1}
                for index in range(2)
            ]
            triples = [
                {"source_index": index, "source_line": index + 1, "triple_index": 0,
                 "subject": "milling", "relation": "Tool", "object": "cutter",
                 "subject_verbatim": True, "object_verbatim": True}
                for index in range(2)
            ]
            write_jsonl(batch / "records.jsonl", records)
            write_jsonl(batch / "triples.jsonl", triples)
            (batch / "summary.json").write_text(
                json.dumps({"total_lines": 2, "canonical_triples": 2,
                            "pending_lines": 0, "pipeline_error_lines": 0}), encoding="utf-8")

            source = load_source(batch, False)
            first = ingest(source, schema, database)
            self.assertEqual(first["graph"], {"datasets": 1, "documents": 2,
                                              "entities": 2, "edges": 1, "occurrences": 2})
            self.assertEqual(neighbors(database, "milling", 5)["neighbors"][0]["evidence_count"], 2)
            self.assertEqual(ingest(source, schema, database)["graph"], first["graph"])

            repeated_position = replace(source, occurrences=source.occurrences + [source.occurrences[0]])
            with self.assertRaises(sqlite3.IntegrityError):
                ingest(repeated_position, schema, database)
            self.assertEqual(stats(database)["graph"], first["graph"])

            triples[1]["object"] = "new cutter"
            write_jsonl(batch / "triples.jsonl", triples)
            replaced = ingest(load_source(batch, False), schema, database)
            self.assertEqual(replaced["graph"]["edges"], 2)
            self.assertEqual(replaced["graph"]["occurrences"], 2)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_latest_stage_and_schema_error_leave_existing_graph_intact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            schema = root / "schema.csv"
            database = root / "graph.sqlite"
            schema.write_text("材料,零件由材料制成。\n", encoding="utf-8")
            for iteration, relation in ((0, "错误关系"), (1, "材料")):
                folder = run / f"iter{iteration}"
                folder.mkdir(parents=True)
                (folder / "result_at_each_stage.json").write_text(json.dumps([{
                    "index": 0, "input_text": "零件由钢制成。",
                    "schema_canonicalizaiton": [["零件", relation, "钢"], None],
                }], ensure_ascii=False), encoding="utf-8")
            source = load_source(run, False)
            self.assertEqual(source.path.parent.name, "iter1")
            result = ingest(source, schema, database)
            self.assertEqual(result["graph"]["occurrences"], 1)
            self.assertEqual(stats(database)["graph"]["edges"], 1)
            with self.assertRaisesRegex(ValueError, "outside target schema"):
                ingest(load_source(run / "iter0", False), schema, database,
                       dataset_id=result["dataset_id"])
            self.assertEqual(stats(database)["graph"]["edges"], 1)

    def test_canonical_text_and_partial_batch_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.csv"
            schema.write_text("Tool,The process uses the tool.\n", encoding="utf-8")
            text_file = root / "canon_kg.txt"
            text_file.write_text("[['milling', 'Tool', 'cutter']]\n[]", encoding="utf-8")
            result = ingest(load_source(text_file, False), schema, root / "graph.sqlite")
            self.assertEqual(result["graph"]["documents"], 2)
            self.assertEqual(result["graph"]["occurrences"], 1)
            batch = root / "batch"
            batch.mkdir()
            write_jsonl(batch / "records.jsonl", [{
                "index": 0, "source_line": 1, "input_text": "milling uses cutter",
                "status": "pending", "flags": ["pending"], "canonical_count": 0,
            }])
            write_jsonl(batch / "triples.jsonl", [])
            (batch / "summary.json").write_text(json.dumps({
                "total_lines": 1, "canonical_triples": 0,
                "pending_lines": 1, "pipeline_error_lines": 0,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pending or error"):
                load_source(batch, False)
            self.assertEqual(len(load_source(batch, True).documents), 1)


if __name__ == "__main__":
    unittest.main()
