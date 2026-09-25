"""Build a persistent property graph from MPKG canonical triples.

The EDC output is the source of facts. This module stores directed, typed
edges and keeps every source occurrence so repeated facts remain traceable.
No model is loaded during graph construction.
"""

import argparse
import ast
import csv
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Document:
    index: int
    line: int
    text: str | None
    status: str
    flags: list[str]


@dataclass(frozen=True)
class Occurrence:
    source_index: int
    source_line: int
    triple_index: int
    subject: str
    relation: str
    object: str
    subject_verbatim: bool | None = None
    object_verbatim: bool | None = None


@dataclass(frozen=True)
class SourceData:
    kind: str
    path: Path
    digest: str
    documents: list[Document]
    occurrences: list[Occurrence]


DDL = """
CREATE TABLE IF NOT EXISTS schemas (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS relations (
    schema_id TEXT NOT NULL REFERENCES schemas(id),
    name TEXT NOT NULL,
    definition TEXT NOT NULL,
    PRIMARY KEY (schema_id, name)
);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    schema_id TEXT NOT NULL REFERENCES schemas(id),
    imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    source_index INTEGER NOT NULL,
    source_line INTEGER NOT NULL,
    input_text TEXT,
    status TEXT NOT NULL,
    flags_json TEXT NOT NULL,
    PRIMARY KEY (dataset_id, source_index)
);
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES entities(id),
    schema_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    object_id INTEGER NOT NULL REFERENCES entities(id),
    FOREIGN KEY (schema_id, relation) REFERENCES relations(schema_id, name),
    UNIQUE (subject_id, schema_id, relation, object_id)
);
CREATE TABLE IF NOT EXISTS occurrences (
    dataset_id TEXT NOT NULL,
    source_index INTEGER NOT NULL,
    triple_index INTEGER NOT NULL,
    edge_id INTEGER NOT NULL REFERENCES edges(id),
    subject_verbatim INTEGER,
    object_verbatim INTEGER,
    PRIMARY KEY (dataset_id, source_index, triple_index),
    FOREIGN KEY (dataset_id, source_index)
        REFERENCES documents(dataset_id, source_index) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS edges_by_subject ON edges(subject_id);
CREATE INDEX IF NOT EXISTS edges_by_object ON edges(object_id);
CREATE INDEX IF NOT EXISTS occurrences_by_edge ON occurrences(edge_id);
"""


def digest_files(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_schema(path: Path) -> tuple[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or any(len(row) != 2 or not all(part.strip() for part in row) for row in rows):
        raise ValueError(f"Schema must contain two nonempty CSV fields per row: {path}")
    relations = {name.strip(): definition.strip() for name, definition in rows}
    if len(relations) != len(rows):
        raise ValueError(f"Duplicate relation name in schema: {path}")
    return digest_files(path), relations


def read_jsonl(path: Path) -> list[dict]:
    values = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL row: {path}:{number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object: {path}:{number}")
            values.append(value)
    return values


def valid_index(value) -> bool:
    return type(value) is int and value >= 0


def make_occurrence(row: dict, location: str) -> Occurrence:
    for key in ("source_index", "triple_index"):
        if not valid_index(row.get(key)):
            raise ValueError(f"Invalid {key} at {location}")
    line = row.get("source_line")
    if type(line) is not int or line < 1:
        raise ValueError(f"Invalid source_line at {location}")
    parts = []
    for key in ("subject", "relation", "object"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid {key} at {location}")
        parts.append(value.strip())
    for key in ("subject_verbatim", "object_verbatim"):
        if row.get(key) is not None and type(row[key]) is not bool:
            raise ValueError(f"Invalid {key} at {location}")
    return Occurrence(row["source_index"], line, row["triple_index"],
                      *parts, row.get("subject_verbatim"), row.get("object_verbatim"))


def batch_source(directory: Path, allow_partial: bool) -> SourceData:
    records_path = directory / "records.jsonl"
    triples_path = directory / "triples.jsonl"
    summary_path = directory / "summary.json"
    if not all(path.is_file() for path in (records_path, triples_path, summary_path)):
        raise ValueError(f"Incomplete batch reports in {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if not allow_partial and (summary.get("pending_lines") or summary.get("pipeline_error_lines")):
        raise ValueError("Batch has pending or error lines; pass --allow-partial to import available triples")
    record_rows = read_jsonl(records_path)
    documents = []
    for position, row in enumerate(record_rows):
        if row.get("index") != position or row.get("source_line") != position + 1:
            raise ValueError(f"Unexpected record index at {records_path}:{position + 1}")
        if not isinstance(row.get("input_text"), str):
            raise ValueError(f"Missing input_text at {records_path}:{position + 1}")
        flags = row.get("flags", [])
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
            raise ValueError(f"Invalid flags at {records_path}:{position + 1}")
        documents.append(Document(position, position + 1, row["input_text"],
                                  str(row.get("status", "unknown")), flags))
    occurrences = [make_occurrence(row, f"{triples_path}:{number}")
                   for number, row in enumerate(read_jsonl(triples_path), 1)]
    if summary.get("total_lines") != len(documents) or summary.get("canonical_triples") != len(occurrences):
        raise ValueError("Batch summary counts disagree with records or triples")
    if (summary.get("pending_lines") != sum(item.status == "pending" for item in documents)
            or summary.get("pipeline_error_lines") != sum(item.status == "error" for item in documents)):
        raise ValueError("Batch summary status counts disagree with records")
    counts = {}
    for item in occurrences:
        if item.source_index >= len(documents) or documents[item.source_index].line != item.source_line:
            raise ValueError(f"Triple has no matching source record: {item}")
        counts[item.source_index] = counts.get(item.source_index, 0) + 1
    for index, row in enumerate(record_rows):
        if row.get("canonical_count") != counts.get(index, 0):
            raise ValueError(f"Canonical count disagrees at source line {index + 1}")
    return SourceData("batch", directory, digest_files(records_path, triples_path, summary_path),
                      documents, occurrences)


def stage_source(path: Path) -> SourceData:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a stage-result JSON array: {path}")
    documents, occurrences = [], []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("input_text"), str):
            raise ValueError(f"Invalid stage row {index} in {path}")
        if row.get("index", index) != index:
            raise ValueError(f"Unexpected stage index {index} in {path}")
        triples = row.get("schema_canonicalizaiton")
        if not isinstance(triples, list):
            raise ValueError(f"Missing canonical triples in stage row {index}")
        documents.append(Document(index, index + 1, row["input_text"].rstrip("\r\n"), "checked", []))
        for triple_index, triple in enumerate(triples):
            if triple is None:
                continue
            if not isinstance(triple, list) or len(triple) != 3:
                raise ValueError(f"Invalid canonical triple in stage row {index}, position {triple_index}")
            occurrences.append(make_occurrence({
                "source_index": index, "source_line": index + 1, "triple_index": triple_index,
                "subject": triple[0], "relation": triple[1], "object": triple[2],
            }, f"{path}:row {index}, triple {triple_index}"))
    return SourceData("stage", path, digest_files(path), documents, occurrences)


def canonical_text_source(path: Path) -> SourceData:
    """Read EDC's per-document Python-list format without evaluating code."""
    documents, occurrences = [], []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        try:
            value = ast.literal_eval(line)
        except (SyntaxError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid canonical list at {path}:{index + 1}") from exc
        if not isinstance(value, list):
            raise ValueError(f"Expected a triple list at {path}:{index + 1}")
        triples = [value] if len(value) == 3 and all(isinstance(part, str) for part in value) else value
        documents.append(Document(index, index + 1, None, "unknown", ["source_text_unavailable"]))
        for triple_index, triple in enumerate(triples):
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                raise ValueError(f"Invalid canonical triple at {path}:{index + 1}")
            occurrences.append(make_occurrence({
                "source_index": index, "source_line": index + 1, "triple_index": triple_index,
                "subject": triple[0], "relation": triple[1], "object": triple[2],
            }, f"{path}:{index + 1}"))
    return SourceData("canon_text", path, digest_files(path), documents, occurrences)


def load_source(path: Path, allow_partial: bool) -> SourceData:
    path = path.resolve()
    if path.is_dir():
        if (path / "triples.jsonl").is_file():
            return batch_source(path, allow_partial)
        if (path / "result_at_each_stage.json").is_file():
            return stage_source(path / "result_at_each_stage.json")
        if (path / "canon_kg.txt").is_file():
            return canonical_text_source(path / "canon_kg.txt")
        iterations = [(int(match.group(1)), child / "result_at_each_stage.json")
                      for child in path.iterdir() if child.is_dir()
                      if (match := re.fullmatch(r"iter(\d+)", child.name))
                      and (child / "result_at_each_stage.json").is_file()]
        if iterations:
            return stage_source(max(iterations)[1])
    elif path.name == "triples.jsonl":
        return batch_source(path.parent, allow_partial)
    elif path.name == "result_at_each_stage.json":
        return stage_source(path)
    elif path.name == "canon_kg.txt":
        return canonical_text_source(path)
    raise ValueError("Input must be a batch output directory, triples.jsonl, stage JSON, canon_kg.txt, or run directory")


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def graph_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("datasets", "documents", "entities", "edges", "occurrences")}


def ingest(source: SourceData, schema_path: Path, database: Path,
           dataset_id: str | None = None) -> dict:
    schema_id, relation_definitions = load_schema(schema_path)
    manifest_path = source.path / "manifest.json" if source.kind == "batch" else None
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if schema_id not in manifest.get("files_sha256", {}).values():
            raise ValueError("Selected schema does not match the batch manifest")
    for item in source.occurrences:
        if item.relation not in relation_definitions:
            raise ValueError(f"Relation outside target schema at line {item.source_line}: {item.relation}")
    identity = dataset_id or "source_" + hashlib.sha256(str(source.path).encode("utf-8")).hexdigest()[:20]
    if not identity.strip():
        raise ValueError("Dataset ID must not be empty")
    if database.resolve() in {source.path.resolve(), schema_path.resolve()}:
        raise ValueError("Database path must differ from source and schema paths")
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(database)) as connection:
        connection.executescript(DDL)
        with connection:
            connection.execute("DELETE FROM datasets WHERE id = ?", (identity,))
            connection.execute("INSERT OR IGNORE INTO schemas VALUES (?, ?)",
                               (schema_id, str(schema_path.resolve())))
            connection.executemany("INSERT OR IGNORE INTO relations VALUES (?, ?, ?)",
                                   [(schema_id, name, definition)
                                    for name, definition in relation_definitions.items()])
            connection.execute("INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?)",
                               (identity, str(source.path), source.kind, source.digest, schema_id,
                                datetime.now(timezone.utc).isoformat()))
            connection.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                                   [(identity, item.index, item.line, item.text, item.status,
                                     json.dumps(item.flags, ensure_ascii=False))
                                    for item in source.documents])
            entity_ids = {}
            for item in source.occurrences:
                for name in (item.subject, item.object):
                    if name not in entity_ids:
                        connection.execute("INSERT OR IGNORE INTO entities(name) VALUES (?)", (name,))
                        entity_ids[name] = connection.execute(
                            "SELECT id FROM entities WHERE name = ?", (name,)).fetchone()[0]
                connection.execute(
                    "INSERT OR IGNORE INTO edges(subject_id, schema_id, relation, object_id) "
                    "VALUES (?, ?, ?, ?)",
                    (entity_ids[item.subject], schema_id, item.relation, entity_ids[item.object]),
                )
                edge_id = connection.execute(
                    "SELECT id FROM edges WHERE subject_id = ? AND schema_id = ? "
                    "AND relation = ? AND object_id = ?",
                    (entity_ids[item.subject], schema_id, item.relation,
                     entity_ids[item.object]),
                ).fetchone()[0]
                connection.execute("INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?)",
                                   (identity, item.source_index, item.triple_index, edge_id,
                                    item.subject_verbatim, item.object_verbatim))
            connection.execute("DELETE FROM edges WHERE NOT EXISTS "
                               "(SELECT 1 FROM occurrences WHERE occurrences.edge_id = edges.id)")
            connection.execute("DELETE FROM entities WHERE NOT EXISTS "
                               "(SELECT 1 FROM edges WHERE edges.subject_id = entities.id "
                               "OR edges.object_id = entities.id)")
        issues = connection.execute("PRAGMA foreign_key_check").fetchall()
        if issues:
            raise RuntimeError(f"Graph foreign key check failed: {issues[:3]}")
        return {"database": str(database.resolve()), "dataset_id": identity,
                "source_kind": source.kind, "source_documents": len(source.documents),
                "source_occurrences": len(source.occurrences), "graph": graph_counts(connection)}


def stats(database: Path) -> dict:
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(connect(database)) as connection:
        return {"database": str(database.resolve()), "graph": graph_counts(connection),
                "datasets": [dict(row) for row in connection.execute(
                    "SELECT id, source_kind, source_path, schema_id FROM datasets ORDER BY id")]}


def neighbors(database: Path, entity: str, limit: int) -> dict:
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(connect(database)) as connection:
        rows = connection.execute(
            "SELECT e.id, s.name AS subject, e.relation, o.name AS object, "
            "e.schema_id, COUNT(x.edge_id) AS evidence_count "
            "FROM edges e JOIN entities s ON s.id = e.subject_id "
            "JOIN entities o ON o.id = e.object_id "
            "JOIN occurrences x ON x.edge_id = e.id "
            "WHERE s.name = ? OR o.name = ? GROUP BY e.id "
            "ORDER BY evidence_count DESC, e.id LIMIT ?", (entity, entity, limit),
        ).fetchall()
        result = []
        for row in rows:
            evidence = [dict(item) for item in connection.execute(
                "SELECT x.dataset_id, d.source_line, x.triple_index, d.input_text "
                "FROM occurrences x JOIN documents d "
                "ON d.dataset_id = x.dataset_id AND d.source_index = x.source_index "
                "WHERE x.edge_id = ? ORDER BY x.dataset_id, d.source_line LIMIT 3", (row["id"],))]
            result.append({**dict(row), "evidence": evidence})
        return {"entity": entity, "neighbors": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("ingest", help="Build or replace one dataset in a SQLite graph")
    build.add_argument("--input", required=True, type=Path)
    build.add_argument("--schema", required=True, type=Path)
    build.add_argument("--db", required=True, type=Path)
    build.add_argument("--dataset-id", help="Stable name when replacing a previous dataset import")
    build.add_argument("--allow-partial", action="store_true")
    summary = commands.add_parser("stats", help="Inspect persisted graph counts")
    summary.add_argument("--db", required=True, type=Path)
    lookup = commands.add_parser("neighbors", help="Find edges and source evidence for an entity")
    lookup.add_argument("--db", required=True, type=Path)
    lookup.add_argument("--entity", required=True)
    lookup.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "ingest":
        result = ingest(load_source(args.input, args.allow_partial), args.schema, args.db,
                        args.dataset_id)
    elif args.command == "stats":
        result = stats(args.db)
    else:
        if args.limit < 1:
            parser.error("--limit must be positive")
        result = neighbors(args.db, args.entity, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
