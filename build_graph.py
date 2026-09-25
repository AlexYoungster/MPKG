"""Build a persistent property graph from MPKG canonical triples.

The EDC output is the source of facts. This module stores directed, typed
edges and keeps every source occurrence so repeated facts remain traceable.
No model is loaded during graph construction.

知识图谱构建模块（抽取产物 → 持久属性图）。

职责边界：本模块只负责"存储建图"，不做任何抽取——事实来源是 EDC 抽取流水线
（OIE → Schema Definition → Schema Canonicalization）输出的规范三元组，
构建过程不加载任何模型。

核心设计：
- 图以 SQLite 持久化：实体节点 + 有向带类型边（由 schema_id, relation 限定）；
- 事实去重但来源不丢：边层按 (subject, schema, relation, object) 唯一去重，
  全部来源记录保留在 occurrences 溯源表中，任何一条边都可回溯到原文出处；
- 严格校验：schema 配套性（manifest 指纹比对）、行号/计数多重对账、
  关系必须落在目标模式内，脏数据一律在入口报错拒绝，绝不静默跳过。
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


# 一条输入文档的元信息：源文件行号、原文（可缺失）、处理状态（checked/pending/error 等）与标记列表。
@dataclass(frozen=True)
class Document:
    index: int
    line: int
    text: str | None
    status: str
    flags: list[str]


# 一次三元组出现（溯源最小单元）：定位到来源文档 (source_index, source_line) 与
# 该文档内的三元组序号；verbatim 标志记录头/尾实体是否为原文逐字出现（None 表示未知）。
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


# 任一来源格式解析后的统一中间表示：kind（batch/stage/canon_text）、来源路径、
# 内容 SHA-256 哈希表明身份，以及解析出的全部文档与三元组出现记录。
@dataclass(frozen=True)
class SourceData:
    kind: str
    path: Path
    digest: str
    documents: list[Document]
    occurrences: list[Occurrence]


# 图库表结构（SQLite 属性图）：
# - schemas/relations：目标模式（关系名→定义），约束图中每条边的关系都可解释；
# - datasets/documents：数据集与输入文档，保留行号、状态与 flags；
# - entities：实体节点表，name 唯一即实现同名实体合并（本项目唯一的实体对齐手段）；
# - edges：有向带类型边，(subject, schema_id, relation, object) 四元组唯一去重；
# - occurrences：溯源表，保留每条边的全部出现记录，删除数据集时随级联清理；
# - 三个索引分别加速"按头实体查边 / 按尾实体查边 / 按边查证据"三类高频查询。
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
    # 按给定顺序拼接多个文件内容计算 SHA-256，作为数据集/模式的可鉴别标识。
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_schema(path: Path) -> tuple[str, dict[str, str]]:
    # 读取目标模式 CSV（每行"关系名,定义"，兼容 BOM）。校验：每行必须恰好两列
    # 且均非空、关系名不得重复。返回 (schema 标识, {关系名: 定义})。
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or any(len(row) != 2 or not all(part.strip() for part in row) for row in rows):
        raise ValueError(f"Schema must contain two nonempty CSV fields per row: {path}")
    relations = {name.strip(): definition.strip() for name, definition in rows}
    if len(relations) != len(rows):
        raise ValueError(f"Duplicate relation name in schema: {path}")
    return digest_files(path), relations


def read_jsonl(path: Path) -> list[dict]:
    # 严格 JSONL 读取：空行或非 JSON 对象的行直接报错（宁可失败，不静默跳过）。
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
    # 严格校验：必须是 int 且非负。用 type() 而非 isinstance，可同时排除
    # bool（bool 是 int 的子类，True/False 不应冒充索引）。
    return type(value) is int and value >= 0


def make_occurrence(row: dict, location: str) -> Occurrence:
    # 把一行三元组记录做字段级校验后构造为 Occurrence：
    # - source_index/triple_index 必须为非负 int，source_line 从 1 起；
    # - subject/relation/object 必须为非空字符串并去首尾空白；
    # - subject_verbatim/object_verbatim 只能是 bool 或缺省 None。
    # 任何字段非法都在入口报错，阻止脏数据流入图库。
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
    """解析 batch 报告目录（records.jsonl + triples.jsonl + summary.json）并做多重对账。"""
    records_path = directory / "records.jsonl"
    triples_path = directory / "triples.jsonl"
    summary_path = directory / "summary.json"
    # 三件套缺一不可。
    if not all(path.is_file() for path in (records_path, triples_path, summary_path)):
        raise ValueError(f"Incomplete batch reports in {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    # 存在 pending/error 行时默认拒绝导入，仅显式 --allow-partial 才放行。
    if not allow_partial and (summary.get("pending_lines") or summary.get("pipeline_error_lines")):
        raise ValueError("Batch has pending or error lines; pass --allow-partial to import available triples")
    record_rows = read_jsonl(records_path)
    documents = []
    # 记录必须按 index 从 0 连续编号，且 source_line 与之对应（从 1 起）。
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
    # 三元组逐行经 make_occurrence 校验后转为出现记录。
    occurrences = [make_occurrence(row, f"{triples_path}:{number}")
                   for number, row in enumerate(read_jsonl(triples_path), 1)]
    # 对账 1：summary 声明的文档数/三元组总数必须与实际行数一致。
    if summary.get("total_lines") != len(documents) or summary.get("canonical_triples") != len(occurrences):
        raise ValueError("Batch summary counts disagree with records or triples")
    # 对账 2：summary 声明的 pending/error 行数与 records 实际状态分布一致。
    if (summary.get("pending_lines") != sum(item.status == "pending" for item in documents)
            or summary.get("pipeline_error_lines") != sum(item.status == "error" for item in documents)):
        raise ValueError("Batch summary status counts disagree with records")
    # 对账 3：每条三元组必须能对应到一篇文档（source_index 与 source_line 双向核对），
    # 同时按文档聚合三元组数。
    counts = {}
    for item in occurrences:
        if item.source_index >= len(documents) or documents[item.source_index].line != item.source_line:
            raise ValueError(f"Triple has no matching source record: {item}")
        counts[item.source_index] = counts.get(item.source_index, 0) + 1
    # 对账 4：records 中每篇文档声明的 canonical_count 与实际三元组数一致。
    for index, row in enumerate(record_rows):
        if row.get("canonical_count") != counts.get(index, 0):
            raise ValueError(f"Canonical count disagrees at source line {index + 1}")
    return SourceData("batch", directory, digest_files(records_path, triples_path, summary_path),
                      documents, occurrences)


def stage_source(path: Path) -> SourceData:
    # 解析 EDC 分阶段结果 result_at_each_stage.json：三元组取自
    # schema_canonicalizaiton 字段（沿用 EDC 的原始拼写）。该格式无法区分
    # 处理状态，文档统一记为 checked。
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
            if triple is None:  # 该次标准化失败（未映射到规范关系）的三元组，跳过不入图
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
    # 解析 canon_kg.txt：每行一个 Python 列表字面量。用 ast.literal_eval 而非
    # eval——只解析字面量、绝不执行代码；兼容单条三元组 [s,r,o] 与三元组列表
    # [[s,r,o],...] 两种行格式。该格式不含原文：文档文本记 None、状态 unknown，
    # 并打 source_text_unavailable 标记。
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
    # 输入自动识别调度器。目录内按优先级依次探测：batch 目录（含 triples.jsonl）
    # > result_at_each_stage.json > canon_kg.txt > 多轮迭代目录（iter0/iter1/...
    # 中取 iter 编号最大的一轮，即最新结果）。也可直接传入上述三种文件之一的路径。
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
    # 行以 sqlite3.Row 返回（可按列名取值）。SQLite 默认不启用外键约束，
    # 必须显式执行 PRAGMA foreign_keys = ON，否则表定义中的 REFERENCES 形同虚设。
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def graph_counts(connection: sqlite3.Connection) -> dict[str, int]:
    # 返回五张核心表的行数统计，用于建图结果汇报与 stats 检查。
    return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("datasets", "documents", "entities", "edges", "occurrences")}


def ingest(source: SourceData, schema_path: Path, database: Path,
           dataset_id: str | None = None) -> dict:
    """核心建图方法：把一份解析后的来源导入（或整体替换）到 SQLite 图库。"""
    # 步骤 1：加载目标模式。若来源为 batch 且带 manifest.json，校验所选 schema
    # 的指纹确实登记在 manifest 中，防止"模式与数据不配套"的错配导入。
    schema_id, relation_definitions = load_schema(schema_path)
    manifest_path = source.path / "manifest.json" if source.kind == "batch" else None
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if schema_id not in manifest.get("files_sha256", {}).values():
            raise ValueError("Selected schema does not match the batch manifest")
    # 步骤 2：入图前校验每条三元组的关系都在目标模式内：图中不允许出现
    # 模式无法解释的边（batch 上游已保证，此处对其他来源兜底）。
    for item in source.occurrences:
        if item.relation not in relation_definitions:
            raise ValueError(f"Relation outside target schema at line {item.source_line}: {item.relation}")
    # 步骤 3：确定数据集标识。未指定时取来源路径 SHA-256 前 20 位，同一来源
    # 重复导入会命中同一 identity，走"整体替换"而非重复累积。
    identity = dataset_id or "source_" + hashlib.sha256(str(source.path).encode("utf-8")).hexdigest()[:20]
    if not identity.strip():
        raise ValueError("Dataset ID must not be empty")
    # 数据库文件不得与来源/模式文件同路径，避免建库时覆盖输入。
    if database.resolve() in {source.path.resolve(), schema_path.resolve()}:
        raise ValueError("Database path must differ from source and schema paths")
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(database)) as connection:
        connection.executescript(DDL)
        # 步骤 4：以下写入在单个事务内完成，任一步失败则整体回滚。
        with connection:
            # 步骤 5：替换语义：先删除同 id 旧数据集（其 occurrences 随级联删除）；
            # schema/relations 用 INSERT OR IGNORE，使多个数据集可共享同一模式。
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
            # 步骤 6：实体/边去重入库，occurrences 保留全部出现记录——
            # 同一事实出现 N 次只产生 1 条边（UNIQUE 约束 + INSERT OR IGNORE）
            # 与 N 条溯源记录：图被去重，但事实来源不丢失。
            entity_ids = {}
            for item in source.occurrences:
                # 头/尾实体先按名字去重入库（entities.name UNIQUE 即同名合并，
                # 也是本项目唯一的实体对齐手段），并缓存 id 避免重复查询。
                for name in (item.subject, item.object):
                    if name not in entity_ids:
                        connection.execute("INSERT OR IGNORE INTO entities(name) VALUES (?)", (name,))
                        entity_ids[name] = connection.execute(
                            "SELECT id FROM entities WHERE name = ?", (name,)).fetchone()[0]
                # 边按 (subject, schema, relation, object) 四元组去重写入。
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
                # 溯源记录挂到对应边：数据集 + 文档行号 + 三元组序号 → edge_id，
                # 并保存实体逐字出现标志，支持回溯到原文。
                connection.execute("INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?)",
                                   (identity, item.source_index, item.triple_index, edge_id,
                                    item.subject_verbatim, item.object_verbatim))
            # 步骤 7：清理孤儿数据——删除没有任何出现记录的边，以及不再参与
            # 任何边的实体（替换数据集被级联清理后可能残留）。
            connection.execute("DELETE FROM edges WHERE NOT EXISTS "
                               "(SELECT 1 FROM occurrences WHERE occurrences.edge_id = edges.id)")
            connection.execute("DELETE FROM entities WHERE NOT EXISTS "
                               "(SELECT 1 FROM edges WHERE edges.subject_id = entities.id "
                               "OR edges.object_id = entities.id)")
        # 步骤 8：外键完整性终检——失败即抛错，绝不产出引用不一致的图库。
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
