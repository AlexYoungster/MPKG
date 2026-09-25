"""Minimal source-grounded Graph RAG over the SQLite graph built by MPKG."""

import os
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


def open_graph(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Graph database not found: {path}")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _question_terms(question: str) -> set[str]:
    return {part.casefold() for part in re.findall(r"[\w#.+-]{3,}", question)}


CATEGORY_RELATIONS = {
    "material", "tool", "equipment", "grinding wheel", "abrasive", "coolant",
    "operation", "control method", "dressing method",
    "材料", "使用刀具", "使用设备", "使用砂轮", "冷却方式", "加工工序", "加工方法",
}
QUANTITY_VALUE = re.compile(
    r"^\s*[≤≥<>~]?(?:约\s*)?\d+(?:\.\d+)?\s*"
    r"(?:mm|cm|µm|μm|m/min|mm/min|mm/rev|mm/r|rpm|r/min|l/min|ml/min|"
    r"bar|mpa|pa|kw|w|v|a|hz|°c|℃|%)\b",
    re.IGNORECASE,
)


def quantity_type_warning(relation: str, value: str) -> bool:
    return relation.casefold() in CATEGORY_RELATIONS and bool(QUANTITY_VALUE.match(value))


class GraphRetriever:
    QUERY_TYPES = {"document_filter", "relation_filter", "path", "text_search"}

    def __init__(self, database: Path, *, max_documents=3, max_plan_documents=20,
                 max_catalog_values=200):
        self.database = Path(database).resolve()
        with closing(open_graph(self.database)) as connection:
            connection.execute("SELECT 1 FROM occurrences LIMIT 1").fetchone()
        self.max_documents = max_documents
        self.max_plan_documents = max_plan_documents
        self.max_catalog_values = max_catalog_values

    @staticmethod
    def _key(row) -> tuple[str, int]:
        return row["dataset_id"], row["source_index"]

    def _entity_names(self) -> list[str]:
        with closing(open_graph(self.database)) as connection:
            return [row[0] for row in connection.execute("SELECT name FROM entities")]

    def _anchor_documents(self, entity: str) -> list[tuple[str, int]]:
        with closing(open_graph(self.database)) as connection:
            rows = connection.execute(
                "SELECT DISTINCT x.dataset_id, x.source_index FROM occurrences x "
                "JOIN edges e ON e.id = x.edge_id "
                "JOIN entities s ON s.id = e.subject_id "
                "JOIN entities o ON o.id = e.object_id "
                "WHERE s.name = ? OR o.name = ? ORDER BY x.dataset_id, x.source_index",
                (entity, entity),
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def _relation_names_in_question(self, question: str) -> list[str]:
        with closing(open_graph(self.database)) as connection:
            labels = [row[0] for row in connection.execute("SELECT DISTINCT name FROM relations")]
        return [name for name in labels if len(name) >= 2
                and name.casefold() in question.casefold()]

    def relation_names(self) -> list[str]:
        """Return the canonical relation vocabulary used by the current graph."""
        with closing(open_graph(self.database)) as connection:
            return [row[0] for row in connection.execute(
                "SELECT name FROM relations ORDER BY name")]

    def relation_descriptions(self) -> dict[str, str]:
        """Return schema definitions so the agent can select relations by meaning."""
        with closing(open_graph(self.database)) as connection:
            return {row[0]: row[1] for row in connection.execute(
                "SELECT name, definition FROM relations ORDER BY name")}

    def entity_names(self, *, limit: int | None = None) -> list[str]:
        """Return graph entity names for planner validation and diagnostics."""
        sql = "SELECT name FROM entities ORDER BY name"
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        else:
            params = ()
        with closing(open_graph(self.database)) as connection:
            return [row[0] for row in connection.execute(sql, params)]

    def _fact_rows(self) -> list[dict]:
        """Load occurrence-scoped facts for bounded, deterministic plan execution."""
        with closing(open_graph(self.database)) as connection:
            rows = connection.execute(
                "SELECT x.dataset_id, x.source_index, x.triple_index, "
                "s.name AS subject, e.relation, o.name AS object, "
                "x.subject_verbatim, x.object_verbatim "
                "FROM occurrences x JOIN edges e ON e.id = x.edge_id "
                "JOIN entities s ON s.id = e.subject_id "
                "JOIN entities o ON o.id = e.object_id "
                "ORDER BY x.dataset_id, x.source_index, x.triple_index"
            ).fetchall()
        facts = []
        for row in rows:
            fact = dict(row)
            fact["type_warning"] = quantity_type_warning(fact["relation"], fact["object"])
            facts.append(fact)
        return facts

    def _resolve_entity(self, value: str) -> list[str]:
        value = str(value or "").strip().casefold()
        if not value:
            return []
        names = self.entity_names()
        exact = [name for name in names if name.casefold() == value]
        if exact:
            return exact
        # Planner output often uses a descriptive fragment instead of the full
        # canonical name. Return only unique/short candidate sets to avoid a
        # broad term silently becoming a cross-document join.
        contains = [name for name in names if len(value) >= 3 and value in name.casefold()]
        return contains[:8]

    def _relation_matches(self, relation: str, requested: str) -> bool:
        relation = relation.casefold()
        requested = str(requested or "").strip().casefold()
        if not requested:
            return False
        if relation == requested:
            return True
        # Permit a planner to use a natural-language relation phrase while
        # still requiring a canonical relation label to be present in SQLite.
        return requested in relation or relation in requested

    def _object_matches(self, value: str, terms: list[str]) -> bool:
        value = value.casefold()
        return bool(terms) and any(str(term).strip().casefold() in value for term in terms if str(term).strip())

    def _catalogue_relation(self, relation: str, *, limit: int, offset: int) -> dict:
        """Return distinct objects for one relation with compact source references."""
        names = self.relation_names()
        matched = next((name for name in names if name.casefold() == str(relation).strip().casefold()), None)
        if matched is None:
            return {"mode": "agent_catalogue", "relation": relation, "catalogs": [],
                    "candidate_values": 0, "returned_values": 0, "offset": offset,
                    "limit": limit, "truncated": False,
                    "error": "relation is not in the graph schema"}
        with closing(open_graph(self.database)) as connection:
            rows = connection.execute(
                "SELECT o.name AS value, COUNT(*) AS occurrences "
                "FROM occurrences x JOIN edges e ON e.id = x.edge_id "
                "JOIN entities o ON o.id = e.object_id "
                "WHERE e.relation = ? GROUP BY o.name ORDER BY o.name",
                (matched,),
            ).fetchall()
            total = len(rows)
            page = rows[offset:offset + limit]
            catalogs = []
            for index, row in enumerate(page, start=offset + 1):
                source_rows = connection.execute(
                    "SELECT DISTINCT x.dataset_id, x.source_index, d.source_line, "
                    "d.input_text, s.name AS subject "
                    "FROM occurrences x JOIN edges e ON e.id = x.edge_id "
                    "JOIN entities s ON s.id = e.subject_id "
                    "JOIN entities o ON o.id = e.object_id "
                    "JOIN documents d ON d.dataset_id = x.dataset_id "
                    "AND d.source_index = x.source_index "
                    "WHERE e.relation = ? AND o.name = ? "
                    "ORDER BY d.source_line LIMIT 3",
                    (matched, row["value"]),
                ).fetchall()
                outgoing = [item[0] for item in connection.execute(
                    "SELECT DISTINCT e2.relation FROM edges e2 "
                    "JOIN entities s2 ON s2.id = e2.subject_id "
                    "JOIN entities o2 ON o2.id = e2.object_id "
                    "WHERE s2.name = ? ORDER BY e2.relation LIMIT 12",
                    (row["value"],),
                ).fetchall()]
                catalogs.append({
                    "id": f"C{index}",
                    "relation": matched,
                    "value": row["value"],
                    "occurrences": row["occurrences"],
                    "outgoing_relations": outgoing,
                    "sources": [dict(source) for source in source_rows],
                })
        return {"mode": "agent_catalogue", "relation": matched,
                "catalogs": catalogs, "candidate_values": total,
                "returned_values": len(catalogs), "offset": offset,
                "limit": limit, "truncated": offset + len(catalogs) < total,
                "error": None}

    def _generic_query(self, plan: dict) -> dict:
        """Interpret a model-authored read-only graph query expression.

        This is intentionally a small query language rather than a list of
        question-specific tools. New question types are expressed by changing
        the plan (match, path, aggregate, union), while SQLite access remains
        parameterized and bounded.
        """
        if not isinstance(plan, dict):
            return {"mode": "agent_graph_query", "documents": [], "facts": [],
                    "catalogs": [], "candidate_documents": 0,
                    "error": "query must be a JSON object"}
        kind = str(plan.get("kind", plan.get("operation", "documents"))).casefold()
        if kind in {"aggregate", "catalogue", "group", "group_by"}:
            relation = plan.get("relation")
            if not relation:
                return {"mode": "agent_graph_query", "documents": [], "facts": [],
                        "catalogs": [], "candidate_documents": 0,
                        "error": "aggregate query requires relation"}
            try:
                limit = min(self.max_catalog_values, max(1, int(plan.get("limit", 100))))
            except (TypeError, ValueError):
                limit = min(self.max_catalog_values, 100)
            try:
                offset = max(0, int(plan.get("offset", 0)))
            except (TypeError, ValueError):
                offset = 0
            result = self._catalogue_relation(relation, limit=limit, offset=offset)
            result["mode"] = "agent_graph_query"
            result["query_kind"] = "aggregate"
            return result
        if kind in {"path", "traverse", "walk"}:
            query = {"type": "path", **plan}
        elif kind in {"documents", "document", "facts", "match", "filter",
                      "match/documents", "documents/match", "document_filter"}:
            query = {"type": "document_filter", **plan}
            match = plan.get("match", {})
            if isinstance(match, dict):
                query.update(match)
            if "relation" not in query and plan.get("relations") is None:
                query.pop("relation", None)
        elif kind == "union":
            parts = plan.get("queries", [])
            if not isinstance(parts, list) or len(parts) > 8:
                return {"mode": "agent_graph_query", "documents": [], "facts": [],
                        "catalogs": [], "candidate_documents": 0,
                        "error": "union requires at most 8 subqueries"}
            results = [self._generic_query(part) for part in parts if isinstance(part, dict)]
            return self.merge_retrievals(*results, answer_mode=plan.get("answer_mode", "list"))
        else:
            return {"mode": "agent_graph_query", "documents": [], "facts": [],
                    "catalogs": [], "candidate_documents": 0,
                    "error": f"unsupported query kind: {kind}"}
        keys, paths = self._keys_for_query(query)
        try:
            limit = min(self.max_plan_documents, max(1, int(plan.get("limit", self.max_plan_documents))))
        except (TypeError, ValueError):
            limit = self.max_plan_documents
        try:
            offset = max(0, int(plan.get("offset", 0)))
        except (TypeError, ValueError):
            offset = 0
        total = len(keys)
        selected = keys[offset:offset + limit]
        result = self._format_keys(selected, mode="agent_graph_query",
                                   candidate_count=total, max_documents=limit,
                                   paths=paths)
        result["query_kind"] = kind
        result["offset"] = offset
        result["limit"] = limit
        result["truncated"] = offset + len(selected) < total
        return result

    def _keys_for_query(self, query: dict) -> tuple[list[tuple[str, int]], list[list[dict]]]:
        """Execute one validated read-only query against occurrence facts.

        The planner can request document filters, relation/object filters, text
        searches, or bounded same-document paths. It cannot submit SQL.
        """
        query_type = query.get("type", "document_filter")
        if query_type not in self.QUERY_TYPES:
            return [], []
        facts = self._fact_rows()
        grouped: dict[tuple[str, int], list[dict]] = {}
        for fact in facts:
            grouped.setdefault((fact["dataset_id"], fact["source_index"]), []).append(fact)
        keys: list[tuple[str, int]] = []
        paths: list[list[dict]] = []

        if query_type == "path":
            starts = []
            for value in query.get("start_entities", []):
                starts.extend(self._resolve_entity(value))
            if not starts and query.get("start_entity"):
                starts = self._resolve_entity(query.get("start_entity"))
            relation_path = [str(item).strip() for item in query.get("relation_path", [])
                             if str(item).strip()][:3]
            if not starts or not relation_path:
                return [], []
            for key, row_facts in grouped.items():
                current = [{"name": name, "facts": []} for name in starts]
                for relation in relation_path:
                    next_nodes = []
                    for node in current:
                        matches = [fact for fact in row_facts
                                   if fact["subject"].casefold() == node["name"].casefold()
                                   and self._relation_matches(fact["relation"], relation)]
                        for fact in matches:
                            next_nodes.append({"name": fact["object"],
                                               "facts": node["facts"] + [fact]})
                    current = next_nodes
                    if not current:
                        break
                for node in current:
                    if node["facts"]:
                        keys.append(key)
                        paths.append(node["facts"])
                        break
            return list(dict.fromkeys(keys)), paths

        requested_entity = query.get("entity") or query.get("anchor")
        entity_names = self._resolve_entity(requested_entity) if requested_entity else []
        relations = query.get("relations", [])
        if isinstance(relations, str):
            relations = [relations]
        relations = [str(item) for item in relations if str(item).strip()]
        relation = query.get("relation")
        if relation:
            relations.append(str(relation))
        object_terms = query.get("object_terms", query.get("objects", []))
        if isinstance(object_terms, str):
            object_terms = [object_terms]
        text_terms = query.get("text_terms", query.get("terms", []))
        if isinstance(text_terms, str):
            text_terms = [text_terms]
        for key, row_facts in grouped.items():
            if entity_names and not any(
                    fact["subject"] in entity_names or fact["object"] in entity_names
                    for fact in row_facts):
                continue
            if relations and not any(
                    any(self._relation_matches(fact["relation"], item) for item in relations)
                    for fact in row_facts):
                continue
            if object_terms and not any(self._object_matches(fact["object"], object_terms)
                                        for fact in row_facts):
                continue
            if query_type == "text_search" and text_terms:
                with closing(open_graph(self.database)) as connection:
                    document = connection.execute(
                        "SELECT input_text FROM documents WHERE dataset_id = ? AND source_index = ?",
                        key).fetchone()
                text = (document[0] if document else "").casefold()
                if not all(str(term).casefold() in text for term in text_terms):
                    continue
            keys.append(key)
        return list(dict.fromkeys(keys)), paths

    def _format_keys(self, keys: list[tuple[str, int]], *, mode: str,
                     anchor: str | None = None, candidate_count: int | None = None,
                     ambiguous: bool | None = None, max_documents: int | None = None,
                     matched_relations: list[str] | None = None,
                     paths: list[list[dict]] | None = None) -> dict:
        candidate_count = len(keys) if candidate_count is None else candidate_count
        max_documents = self.max_documents if max_documents is None else max_documents
        keys = keys[:max(1, max_documents)]
        documents, facts = [], []
        for key in keys:
            document, row_facts = self._document_facts(key)
            document["id"] = f"D{len(documents) + 1}"
            documents.append(document)
            for row in row_facts:
                row["id"] = f"E{len(facts) + 1}"
                row["document_id"] = document["id"]
                row["dataset_id"] = document["dataset_id"]
                row["source_line"] = document["source_line"]
                facts.append(row)
        return {"mode": mode, "anchor": anchor, "ambiguous": candidate_count > 1
                if ambiguous is None else ambiguous,
                "candidate_documents": candidate_count,
                "matched_relations": matched_relations or [],
                "documents": documents, "facts": facts,
                "paths": paths or [],
                "evidence_truncated": candidate_count > len(documents)}

    def merge_retrievals(self, *retrieved: dict, answer_mode: str = "single") -> dict:
        """Merge documents from several query rounds without mixing source facts."""
        keys = []
        catalogs = []
        seen_catalogs = set()
        candidate_count = 0
        candidate_values = 0
        catalogue_truncated = False
        for item in retrieved:
            candidate_count = max(candidate_count, int(item.get("candidate_documents", 0) or 0))
            candidate_values = max(candidate_values, int(item.get("candidate_values", 0) or 0))
            catalogue_truncated = catalogue_truncated or bool(
                item.get("catalogs") and item.get("truncated", False))
            for catalog in item.get("catalogs", []):
                identity = (catalog.get("relation"), catalog.get("value"))
                if identity not in seen_catalogs:
                    seen_catalogs.add(identity)
                    catalogs.append(catalog)
            for document in item.get("documents", []):
                key = (document["dataset_id"], document["source_index"])
                if key not in keys:
                    keys.append(key)
        result = self._format_keys(keys, mode="multi_hop",
                                   candidate_count=max(candidate_count, len(keys)),
                                   ambiguous=(answer_mode == "single" and len(keys) > 1),
                                   max_documents=self.max_plan_documents)
        result["answer_mode"] = answer_mode
        result["query_count"] = sum(item.get("query_count", 0) for item in retrieved)
        result["catalogs"] = catalogs
        result["candidate_values"] = max(candidate_values, len(catalogs))
        result["catalogue_truncated"] = catalogue_truncated or candidate_values > len(catalogs)
        result["evidence_truncated"] = result["evidence_truncated"] or result["catalogue_truncated"]
        return result

    def execute_tool(self, tool: str, arguments: dict) -> dict:
        """Run one read-only graph tool selected by the language-model agent."""
        arguments = arguments if isinstance(arguments, dict) else {}
        if tool == "graph_query":
            plan = arguments.get("query", arguments)
            result = self._generic_query(plan)
            result["tool"] = tool
            result["arguments"] = arguments
            return result
        if tool == "catalogue":
            relation = arguments.get("relation", "")
            try:
                limit = min(self.max_catalog_values, max(1, int(arguments.get("limit", 100))))
            except (TypeError, ValueError):
                limit = min(self.max_catalog_values, 100)
            try:
                offset = max(0, int(arguments.get("offset", 0)))
            except (TypeError, ValueError):
                offset = 0
            result = self._catalogue_relation(relation, limit=limit, offset=offset)
            result["tool"] = tool
            result["arguments"] = {key: value for key, value in arguments.items()
                                    if key not in {"limit", "offset"}}
            return result
        if tool == "search_documents":
            query = {"type": "document_filter", **arguments}
        elif tool == "filter_facts":
            query = {"type": "relation_filter", **arguments}
        elif tool == "find_paths":
            query = {"type": "path", **arguments}
        elif tool == "search_text":
            query = {"type": "text_search", **arguments}
        else:
            return {"mode": "invalid_tool", "documents": [], "facts": [],
                    "candidate_documents": 0, "error": "unsupported graph tool"}
        keys, paths = self._keys_for_query(query)
        limit = arguments.get("limit", self.max_plan_documents)
        offset = arguments.get("offset", 0)
        try:
            limit = min(self.max_plan_documents, max(1, int(limit)))
        except (TypeError, ValueError):
            limit = self.max_plan_documents
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        total = len(keys)
        keys = keys[offset:offset + limit]
        result = self._format_keys(keys, mode=f"agent_{tool}",
                                   candidate_count=total, max_documents=limit,
                                   paths=paths)
        result["offset"] = offset
        result["limit"] = limit
        result["returned_documents"] = len(keys)
        result["truncated"] = offset + len(keys) < total
        result["tool"] = tool
        result["arguments"] = {key: value for key, value in arguments.items()
                                if key not in {"limit", "offset"}}
        return result

    def _document_facts(self, key: tuple[str, int]) -> tuple[dict, list[dict]]:
        with closing(open_graph(self.database)) as connection:
            document = connection.execute(
                "SELECT dataset_id, source_index, source_line, input_text, status, flags_json "
                "FROM documents WHERE dataset_id = ? AND source_index = ?", key,
            ).fetchone()
            if document is None:
                raise RuntimeError(f"Missing graph document: {key}")
            facts = [dict(row) for row in connection.execute(
                "SELECT x.triple_index, s.name AS subject, e.relation, o.name AS object, "
                "x.subject_verbatim, x.object_verbatim "
                "FROM occurrences x JOIN edges e ON e.id = x.edge_id "
                "JOIN entities s ON s.id = e.subject_id "
                "JOIN entities o ON o.id = e.object_id "
                "WHERE x.dataset_id = ? AND x.source_index = ? ORDER BY x.triple_index", key,
            )]
        for fact in facts:
            fact["type_warning"] = quantity_type_warning(fact["relation"], fact["object"])
        result = dict(document)
        result["flags"] = json.loads(result.pop("flags_json"))
        return result, facts

    def retrieve(self, question: str, *, entity: str | None = None) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        question = question.strip()
        names = self._entity_names()
        if entity is not None:
            matched = next((name for name in names if name.casefold() == entity.strip().casefold()), None)
            if matched is None:
                return {"mode": "explicit_entity_missing", "anchor": entity, "ambiguous": False,
                        "documents": [], "facts": []}
            anchor = matched
            keys = self._anchor_documents(anchor)
            mode = "explicit_entity"
        else:
            matches = [name for name in names if len(name) >= 2
                       and name.casefold() in question.casefold()]
            matches.sort(key=lambda name: (-len(name), name))
            if matches:
                anchor = matches[0]
                keys = self._anchor_documents(anchor)
                mode = "exact_entity"
            else:
                anchor = None
                keys = []
                mode = "entity_missing"

        relation_names = self._relation_names_in_question(question)
        if len(keys) > 1 and relation_names:
            scored = []
            for key in keys:
                _, row_facts = self._document_facts(key)
                score = sum(fact["relation"] in relation_names for fact in row_facts)
                scored.append((score, key))
            highest = max(score for score, _ in scored)
            if highest > 0:
                keys = [key for score, key in scored if score == highest]
        candidate_count = len(keys)
        ambiguous = candidate_count > 1
        if ambiguous:
            terms = _question_terms(question)
            scored = []
            for key in keys:
                document, _ = self._document_facts(key)
                text = (document["input_text"] or "").casefold()
                score = sum(term in text for term in terms)
                scored.append((score, key))
            scored.sort(key=lambda item: (-item[0], item[1]))
            keys = [key for _, key in scored[:self.max_documents]]

        documents, facts = [], []
        for key in keys:
            document, row_facts = self._document_facts(key)
            document["id"] = f"D{len(documents) + 1}"
            documents.append(document)
            for row in row_facts:
                row["id"] = f"E{len(facts) + 1}"
                row["document_id"] = document["id"]
                row["dataset_id"] = document["dataset_id"]
                row["source_line"] = document["source_line"]
                facts.append(row)
        return {"mode": mode, "anchor": anchor, "ambiguous": ambiguous,
                "candidate_documents": candidate_count,
                "matched_relations": relation_names,
                "documents": documents, "facts": facts}


SYSTEM_PROMPT = """你是机械加工工艺问答助手。无论用户使用何种语言，始终用中文回答；工件英文名、数值和单位可保留原文。
请只依据用户提供的图谱事实和对应原文作答。
图谱事实由模型抽取，原文是核验依据；如两者冲突，以原文为准并指出冲突。
带单位的量值表示参数，不能被当作材料、刀具、设备或冷却介质的名称；若原文只给出类别而未指明具体品种，应明确说明。
只回答问题明确询问的项目，不罗列证据中的其他参数。用两到四句连贯的技术中文作答，保留工件名称、数值和单位的原貌。
对于全库概览问题，可以按语义归纳，但每个具体工艺名称必须原样出现在 Operation 聚合值或原文中，并用对应的 [C#]/[D#] 支撑；不要把工件、材料、刀具、阶段名称或抽取噪声直接当成工艺。类别标题只是归纳标签，不能凭空增加一个图谱中没有的具体工艺。不要机械抄录全部聚合值，也不要在输出达到长度上限时用逗号列表截断。
在结论句末就地标注证据编号，如 [E2]、[D1] 或聚合目录项 [C1]，再用一句话说明出处。引用 C 编号的同一句必须包含该 C 项的原样 value；类别标题不能单独借用一个不相符的 C 编号。不要另附三元组清单、表格或未被询问的工艺建议。
不要把其他文本中同名工序的参数移到当前工件上。
若证据不足、主体不明确或无法区分多个工件，明确说无法确定，并说明缺少什么。不要编造工艺参数或建议。"""


def build_messages(question: str, retrieved: dict) -> list[dict]:
    lines = [f"问题：{question}", "", "以下证据按原始文本分组，同组图谱事实才属于同一次记录："]
    catalogs = retrieved.get("catalogs", [])
    if catalogs:
        lines.append("以下是图谱聚合结果；每个 C 编号代表关系客体的去重值，括号内给出出现次数和该值的出边关系。C 项的完整原文出处保存在 evidence 中，回答时可用 C 编号引用：")
        for catalog in catalogs[:200]:
            outgoing = ",".join(catalog.get("outgoing_relations", [])[:8]) or "无"
            value = str(catalog.get("value", ""))[:100]
            lines.append(f"[{catalog['id']}] {catalog['relation']} = {value}；"
                         f"出现 {catalog['occurrences']} 次；主体出边：{outgoing}")
        if retrieved.get("catalogue_truncated"):
            lines.append("聚合结果仍有未返回的值，不能声称下面的列举已经覆盖全库。")
    for document in retrieved["documents"]:
        lines.append(f"[{document['id']}] 数据集 {document['dataset_id']}，原文第 {document['source_line']} 行："
                     f"{document['input_text'] or '原文未保存'}")
        for fact in retrieved["facts"]:
            if fact["document_id"] == document["id"]:
                if fact["type_warning"]:
                    lines.append(f"[{fact['id']}] 该抽取项的客体是量值，与类别关系的类型不一致；"
                                 "不要把量值作为该类别名称。请按原文判断。")
                else:
                    lines.append(f"[{fact['id']}] {fact['subject']} --{fact['relation']}--> {fact['object']}")
    if retrieved["ambiguous"]:
        if retrieved.get("answer_mode") in {"list", "compare", "aggregate", "sequence"}:
            lines.append("当前证据包含多个原文记录；请按文档编号逐条列举或比较，不能把不同记录的参数合并成一条事实。")
        else:
            lines.append("该实体出现在更多原文中；当前仅列出部分记录。不能据此给出唯一参数时请要求补充工件或来源行号。")
    if retrieved.get("evidence_truncated"):
        lines.append("证据记录被数量上限截断，不能声称已经列举全部匹配记录；如问题要求全量结果，应继续查询或明确说明无法完整覆盖。")
    lines.append("请只回答问题所问；如果是全库概览或列举问题，按聚合结果归纳 4～8 个类别，每类列出若干个聚合结果中原样存在的具体工艺。每个类别只标注 1～3 个确实支持该类别的 [C#]，不要把全部 C 编号堆进答案，也不要让同一个 C 编号支撑不相关的多个类别。优先把作为主体还连接 Tool、Cutting Speed、Spindle Speed、Wheel Speed 等参数关系的值判断为工艺；根据关系客体在原文中的语法角色排除工件、材料、刀具和粗加工阶段名称，不要使用笼统的“加工”代替具体工艺。")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


class QwenGenerator:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", *, cache_root: Path | None = None,
                 offline=False, max_new_tokens=320,
                 max_context_tokens: int | None = None):
        self.model_name = model_name
        self.cache_root = cache_root or Path(os.environ.get(
            "MPKG_MODEL_CACHE", Path(__file__).resolve().parent / ".cache" / "models"))
        self.offline = offline
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.context_limit = None
        self.model = None
        self.tokenizer = None
        self.token_usage = []
        self._generation_calls = 0

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-1.7B 的当前 8-bit 配置需要可用的 CUDA 显卡")
        cache = self.cache_root / "transformers"
        quantization = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0,
                                         llm_int8_has_fp16_weight=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=str(cache), local_files_only=self.offline)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", cache_dir=str(cache),
            quantization_config=quantization, torch_dtype=torch.float16,
            local_files_only=self.offline)
        self.model.eval()
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(model_limit, int) or model_limit < 512:
            model_limit = 4096
        if self.max_context_tokens is not None:
            if self.max_context_tokens < 512:
                raise ValueError("max_context_tokens must be at least 512")
            model_limit = min(model_limit, self.max_context_tokens)
        self.context_limit = model_limit

    def _record_usage(self, *, stage: str, input_tokens: int,
                      output_tokens: int, requested_output_tokens: int,
                      context_limit: int) -> None:
        self._generation_calls += 1
        self.token_usage.append({
            "call": self._generation_calls,
            "stage": stage,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "requested_output_tokens": requested_output_tokens,
            "context_limit": context_limit,
        })

    def generate(self, messages: list[dict], max_new_tokens: int | None = None,
                 *, stage: str = "answer") -> str:
        self._load()
        import torch
        from transformers import GenerationConfig

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        requested_output_tokens = max_new_tokens or self.max_new_tokens
        context_limit = self.context_limit or 4096
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens + requested_output_tokens > context_limit:
            raise ValueError(
                f"上下文长度不足：输入 {input_tokens} + 请求输出 {requested_output_tokens} "
                f"> 模型上限 {context_limit} tokens；请减少证据或降低输出长度"
            )
        generation = GenerationConfig(
            do_sample=False, max_new_tokens=requested_output_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.model.generation_config.eos_token_id,
        )
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=generation)
        new_ids = output[0][inputs["input_ids"].shape[-1]:]
        self._record_usage(stage=stage, input_tokens=input_tokens,
                           output_tokens=int(new_ids.shape[-1]),
                           requested_output_tokens=requested_output_tokens,
                           context_limit=context_limit)
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        answer = re.sub(r"^<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
        return answer

    def translate_for_lookup(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "把用户的机械加工问题改写成一句简短的英文检索句。"
             "只做字面翻译，准确翻译工件和工序名称，保留型号、数值、单位；不要回答问题、猜测任何数值或增添事实。‘是多少’必须保留为英文疑问句。只输出英文句子。"},
            {"role": "user", "content": question},
        ]
        output = self.generate(messages, max_new_tokens=80, stage="translation")
        return output.splitlines()[0].strip().strip('"“”') if output else ""

    def agent_step(self, question: str, observation: str,
                   relation_names: dict[str, str] | list[str], tools: tuple[str, ...]) -> dict:
        """Ask Qwen to select exactly one next graph action."""
        payload = {
            "question": question,
            "observation": observation,
            "canonical_relations": relation_names,
            "tools": list(tools),
        }
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        return _parse_agent_json(self.generate(messages, max_new_tokens=220, stage="agent"))


def _document_source(document: dict) -> str:
    """Return the complete source text used for a document citation."""
    text = (document.get("input_text") or "").strip()
    return text or "原文未保存"


def _document_citation(document: dict) -> str:
    """Format a document reference with its complete source text."""
    return f"[{document['id']}]：“{_document_source(document)}”"


def evidence_explanation(retrieved: dict, citation_ids: list[str]) -> str:
    facts = {fact["id"]: fact for fact in retrieved["facts"]}
    documents = {doc["id"]: doc for doc in retrieved["documents"]}
    catalogs = {item["id"]: item for item in retrieved.get("catalogs", [])}
    selected = [facts[key] for key in dict.fromkeys(citation_ids) if key in facts]
    statements = []
    for fact in selected:
        document = documents.get(fact["document_id"])
        source = _document_citation(document) if document else (
            f"[{fact['document_id']}]：原文未保存"
        )
        statements.append(
            f"[{fact['id']}] 记录“{fact['subject']}—{fact['relation']}—{fact['object']}”，"
            f"对应原文 {source}"
        )
    for key in dict.fromkeys(citation_ids):
        catalog = catalogs.get(key)
        if not catalog:
            continue
        source_text = "；".join(
            f"原文第 {source.get('source_line')} 行：“{source.get('input_text') or '原文未保存'}”"
            for source in catalog.get("sources", [])
        ) or "原文出处未保存"
        statements.append(
            f"[{key}] 聚合记录“{catalog['relation']}—{catalog['value']}”（出现 "
            f"{catalog['occurrences']} 次），对应 {source_text}"
        )
    if not statements and documents:
        first = next(iter(documents.values()))
        statements.append(f"[{first['id']}] 对应原文 {_document_citation(first)}")
    return " ".join(statements)


def prose_answer(raw: str, valid_citations: set[str]) -> tuple[str, list[str]]:
    """Keep the answer paragraph and move redundant triple dumps to evidence JSON."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    answer = paragraphs[0] if paragraphs else ""
    answer = re.sub(r"\[([CDE]\d+)\]",
                    lambda match: match.group(0) if match.group(1) in valid_citations else "",
                    answer)
    cited = [key for key in dict.fromkeys(re.findall(r"\[([CDE]\d+)\]", answer))
             if key in valid_citations]
    answer = re.sub(
        r"\[E\d+\]\s*[^。；;\n]*?--[^。；;\n]*?-->[^。；;\n]*(?:[。；;]|$)",
        "", answer,
    )
    answer = re.sub(r"\s+", " ", answer).strip()
    retained = set(re.findall(r"\[([CDE]\d+)\]", answer))
    missing = [key for key in cited if key not in retained]
    if missing:
        answer += " 依据图谱与原文记录 " + "".join(f"[{key}]" for key in missing) + "。"
    return answer, cited


def compose_catalog_answer(retrieved: dict) -> tuple[str, list[str], str]:
    """Deterministically assemble a catalogue answer from graph aggregates.

    Enumeration, ordering and citation binding are program jobs; the model is
    never asked to reproduce a large catalogue in prose.  This keeps aggregate
    answers complete (no generation-limit truncation) and binds every value
    verbatim from the graph to its own citation id.  Grouping is purely
    structural — values that act as a subject with parameter relations are
    reported as parameter-backed, the rest as name-only records.
    """
    catalogs = retrieved.get("catalogs", [])
    if not catalogs:
        return "", [], ""
    relation = str(catalogs[0].get("relation", ""))
    total_occurrences = sum(int(item.get("occurrences", 0)) for item in catalogs)
    parameterized, name_only = [], []
    for item in catalogs:
        outgoing = [r for r in item.get("outgoing_relations", []) if r != relation]
        (parameterized if outgoing else name_only).append(item)
    lines = [f"当前图谱在关系“{relation}”下共聚合出 {len(catalogs)} 个不同客体值，"
             f"来自 {total_occurrences} 次图谱记录。"]
    cited = []
    if parameterized:
        lines.append(f"一、有参数记录支撑的 {len(parameterized)} 项（名称为图谱原样值，"
                     "括号内为记录次数与作为主体时的参数关系）：")
        for item in parameterized:
            outgoing = [r for r in item.get("outgoing_relations", []) if r != relation]
            suffix = " 等" if len(outgoing) > 8 else ""
            lines.append(f"- {item['value']} [{item['id']}]（{item['occurrences']} 次记录；"
                         f"参数关系：{'、'.join(outgoing[:8])}{suffix}）")
            cited.append(item["id"])
    if name_only:
        lines.append(f"二、仅出现名称、未记录参数的 {len(name_only)} 项（证据较弱，供核对原文）：")
        for item in name_only:
            lines.append(f"- {item['value']} [{item['id']}]（{item['occurrences']} 次记录）")
            cited.append(item["id"])
    if retrieved.get("catalogue_truncated"):
        lines.append("注意：聚合结果未覆盖全部客体值，以上并非完整清单。")
    answer = "\n".join(lines)
    explanation = evidence_explanation(retrieved, cited)
    return answer, cited, explanation


def _drop_mismatched_catalog_citations(answer: str, citations: list[str],
                                       catalogs: list[dict]) -> tuple[str, list[str]]:
    """Remove C citations whose candidate value is absent from their sentence.

    The language model may occasionally attach a valid C number to a nearby
    but different category.  This small grounding check keeps the citation
    list source-accurate while leaving the model's prose intact for review.
    """
    values = {item.get("id"): str(item.get("value", "")).casefold()
              for item in catalogs}
    kept = []
    for citation in citations:
        value = values.get(citation)
        if not value:
            kept.append(citation)
            continue
        token = f"[{citation}]"
        position = answer.find(token)
        if position >= 0:
            starts = [answer.rfind(mark, 0, position) for mark in "。！？!?;\n"]
            start = max(starts, default=-1) + 1
            ends = [answer.find(mark, position + len(token))
                    for mark in "。！？!?;\n"]
            ends = [item for item in ends if item >= 0]
            end = min(ends) if ends else len(answer)
            sentence = answer[start:end]
        else:
            sentence = ""
        # A citation is often placed after the sentence-final punctuation, so
        # include the immediately preceding context as a fallback.
        matches_value = value in sentence.casefold()
        if not matches_value:
            matches_value = value in answer[max(0, position - 300):position].casefold()
        if matches_value:
            kept.append(citation)
        else:
            answer = answer.replace(token, "")
    return re.sub(r"\s+", " ", answer).strip(), kept


AGENT_SYSTEM_PROMPT = """你是机械加工知识图谱代理。你负责规划查询、核验结果并判断证据是否足够。
每轮只输出一个 JSON 对象，不要输出 Markdown、解释或思维过程；不得凭常识补造事实。

唯一工具是 graph_query。它接受你自行组合的只读声明式查询：
{"action":"query","tool":"graph_query","arguments":{"query":{...}}}
query.kind 可用 aggregate（按 relation 聚合客体并给出原文出处）、match/documents（按 entity、relations、object_terms、text_terms 过滤原文）、path/traverse（同一原文内沿 relation_path 多跳）、union（合并子查询）。可组合 limit、offset 等字段。关系名必须来自用户消息中的 canonical_relations（每个名称后附图谱定义）；实体和客体词可用英文图谱名称。

你要根据问题自行发现需要的关系和查询步骤：单条参数查找、跨工件比较、全库列举、分类归纳和多跳追踪都用 graph_query 表达，不要等待程序提供专用工具。宽泛地询问“有哪些工艺/操作”时，选择定义为“工件经历的制造操作”的关系做 aggregate；描述操作先后顺序的关系只用于序列追踪。聚合结果带有值、出现次数、原文和该值作为主体时的出边关系；用这些信息判断它是工艺、工件、材料、刀具、阶段还是噪声。
观察中如果已经有一篇原文和事实直接给出了问题所问的关系、客体或参数，先核对主体是否一致，然后直接 finish sufficient=true；不要因为还存在其他关系就扩大查询范围。只有缺少所问事实、存在冲突或问题要求全库列举时才追加查询。
例如，问题只问全库有哪些具体制造工艺且没有指定实体时，应直接生成：{"action":"query","tool":"graph_query","arguments":{"query":{"kind":"aggregate","relation":"Operation","limit":100}}}。这里 Operation 是关系名，不是 entity；不要把关系名填入 entity，也不要把中文问题词放入 object_terms，除非它确实是图谱客体。
若结果被截断，判断问题是否真的需要其余记录；问题要求“全部/有哪些”时应按 offset 分页，直到你认为覆盖完整。若证据不足、主体冲突或需要另一种视角，主动追加不同查询；同一查询不要重复。

只有你确认问题所需范围已经覆盖时才 finish。finish 必须包含 sufficient（true/false）、answer_mode（single/list/compare/aggregate/sequence），并可包含 confidence、coverage、missing，说明你自己的核验结论：
{"action":"finish","sufficient":true,"answer_mode":"aggregate","confidence":0.9,"coverage":"...","missing":[]}
无法从图谱可靠回答时也 finish，但 sufficient=false，并说明 missing。"""


def _parse_agent_json(raw: str) -> dict:
    """Parse a model action while tolerating a single fenced JSON response."""
    if not raw:
        return {}
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        # Qwen sometimes appends a second sentence or another JSON object.
        # Decode the first complete object instead of using a greedy regex.
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            return value if isinstance(value, dict) else {}
        return {}
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _agent_observation(question: str, retrieved: dict, trace: list[dict]) -> str:
    """Build a compact observation; model reasoning stays outside the API result."""
    lines = [f"问题：{question}", "", "当前图谱证据："]
    documents = retrieved.get("documents", [])
    facts = retrieved.get("facts", [])
    catalogs = retrieved.get("catalogs", [])
    candidate_count = retrieved.get("candidate_documents", len(documents))
    candidate_values = retrieved.get("candidate_values", len(catalogs))
    if candidate_count > len(documents):
        lines.append(f"当前只展示 {len(documents)} / {candidate_count} 条候选记录；证据尚未覆盖全部记录。")
    if not documents:
        lines.append("（尚无命中的原文记录）")
    elif facts:
        lines.append("已有命中的原文和图谱事实；如果其中直接包含问题所问关系与参数，先核对主体后结束，不要扩大查询范围。")
    if catalogs:
        lines.append(f"当前目录包含 {len(catalogs)} / {candidate_values} 个聚合值：")
        for catalog in catalogs[:120]:
            # Planning needs the candidate values and graph roles. Full source
            # sentences stay in evidence for the final citation explanation.
            outgoing = ",".join(catalog.get("outgoing_relations", [])[:8]) or "无"
            value = str(catalog.get("value", ""))[:100]
            lines.append(f"  [{catalog['id']}] {catalog['relation']} = {value} "
                         f"（{catalog['occurrences']} 次；主体出边：{outgoing}）")
        if len(catalogs) > 120:
            lines.append(f"（另有 {len(catalogs) - 120} 个聚合值未展开）")
    for document in documents[:12]:
        text = (document.get("input_text") or "原文未保存").strip()
        lines.append(f"[{document['id']}] 行 {document['source_line']}：{text[:700]}")
        for fact in facts:
            if fact.get("document_id") == document["id"]:
                lines.append(f"  [{fact['id']}] {fact['subject']} --{fact['relation']}--> {fact['object']}")
    if len(documents) > 12:
        lines.append(f"（另有 {len(documents) - 12} 条记录未展开）")
    if trace:
        lines.extend(["", "已执行的工具动作："])
        for item in trace[-8:]:
            arguments = json.dumps(item.get("arguments", {}), ensure_ascii=False,
                                   separators=(",", ":"))
            if item.get("truncated"):
                try:
                    current_offset = int(item.get("arguments", {}).get("offset", 0))
                except (TypeError, ValueError):
                    current_offset = 0
                next_offset = current_offset + item.get("documents", 0)
                suffix = f"，结果还有未返回记录；下一页必须使用 offset={next_offset}"
            else:
                suffix = ""
            if item.get("error"):
                suffix += f"；工具提示：{item['error']}"
            values = item.get("values", 0)
            value_text = f"、{values} 个聚合值" if values else ""
            lines.append(f"- {item['tool']} {arguments}：命中 {item['documents']} 条记录、"
                         f"{item['facts']} 条事实{value_text}{suffix}")
    return "\n".join(lines)


class GraphAgent:
    """Model-directed graph tool loop with bounded, auditable state transitions."""

    # The model sees one extensible graph interface. Legacy names are accepted
    # for existing integrations, but are not advertised as the agent protocol.
    TOOLS = ("graph_query",)
    LEGACY_TOOLS = ("catalogue", "search_documents", "filter_facts", "find_paths", "search_text")

    def __init__(self, retriever: GraphRetriever, generator, *, max_steps: int = 6):
        self.retriever = retriever
        self.generator = generator
        self.max_steps = max(1, max_steps)

    def run(self, question: str, seed: dict | None = None) -> dict:
        evidence = seed or {"documents": [], "facts": [], "candidate_documents": 0}
        rounds = [evidence]
        trace = []
        answer_mode = "single"
        status = "step_limit"
        assessment = {
            "sufficient": False,
            "confidence": None,
            "coverage": "代理尚未完成证据核验",
            "missing": ["agent_finish"],
        }
        seen_queries = set()
        for step in range(self.max_steps):
            observation = _agent_observation(question, evidence, trace)
            decision = self.generator.agent_step(
                question, observation, self.retriever.relation_descriptions(), self.TOOLS)
            action = decision.get("action")
            if action == "finish":
                answer_mode = decision.get("answer_mode", "single")
                if answer_mode not in {"single", "list", "compare", "aggregate", "sequence"}:
                    answer_mode = "single"
                # Sufficiency is the model's verification decision.  Retrieval
                # metadata such as pagination/truncation is shown to the model
                # and is not silently converted into a human-authored verdict.
                sufficient = decision.get("sufficient") is True
                status = "sufficient" if sufficient else "insufficient"
                assessment = {
                    "sufficient": sufficient,
                    "confidence": decision.get("confidence"),
                    "coverage": decision.get("coverage", decision.get("reason", "")),
                    "missing": decision.get("missing", []),
                }
                break
            selected_tool = decision.get("tool")
            if action != "query" or selected_tool not in self.TOOLS + self.LEGACY_TOOLS:
                trace.append({"step": step + 1, "tool": "invalid_action",
                              "arguments": {},
                              "documents": 0, "facts": 0})
                continue
            tool = selected_tool
            arguments = decision.get("arguments", {})
            signature = (tool, json.dumps(arguments, ensure_ascii=False,
                                           sort_keys=True, separators=(",", ":")))
            if signature in seen_queries:
                trace.append({"step": step + 1, "tool": tool, "arguments": arguments,
                              "documents": 0, "facts": 0, "truncated": False,
                              "total_documents": 0,
                              "error": "重复查询；请改变条件或为分页查询增加 offset"})
                continue
            seen_queries.add(signature)
            result = self.retriever.execute_tool(tool, arguments)
            evidence = self.retriever.merge_retrievals(
                evidence, result, answer_mode=answer_mode)
            rounds.append(result)
            trace.append({"step": step + 1, "tool": tool,
                          "arguments": arguments,
                          "documents": len(result.get("documents", [])),
                          "facts": len(result.get("facts", [])),
                          "values": len(result.get("catalogs", [])),
                          "truncated": result.get("truncated", False),
                          "total_documents": result.get("candidate_documents", 0),
                          "error": result.get("error")})
        else:
            status = "step_limit"
            assessment = {
                "sufficient": False,
                "confidence": None,
                "coverage": "达到代理步数上限，未收到 finish 核验结论",
                "missing": ["agent_finish"],
            }
        evidence["answer_mode"] = answer_mode
        evidence["agent_status"] = status
        evidence["agent_assessment"] = assessment
        evidence["agent_trace"] = trace
        evidence["agent_rounds"] = len(rounds)
        return evidence


def _call_generator(generator, messages: list[dict], *, stage: str,
                    max_new_tokens: int | None = None) -> str:
    """Call generators with stage accounting while keeping test integrations compatible."""
    kwargs = {"stage": stage}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    try:
        return generator.generate(messages, **kwargs)
    except TypeError:
        # Small test/demonstration generators often implement only generate(messages).
        kwargs.pop("stage", None)
        try:
            return generator.generate(messages, **kwargs)
        except TypeError:
            if max_new_tokens is not None:
                return generator.generate(messages)
            raise


def _usage_snapshot(generator, start: int = 0) -> tuple[list[dict], dict[str, int], int | None]:
    usage = list(getattr(generator, "token_usage", [])[start:])
    totals = {
        "input_tokens": sum(int(item.get("input_tokens", 0)) for item in usage),
        "output_tokens": sum(int(item.get("output_tokens", 0)) for item in usage),
        "total_tokens": sum(int(item.get("total_tokens", 0)) for item in usage),
    }
    return usage, totals, getattr(generator, "context_limit", None)


class GraphRAG:
    def __init__(self, retriever: GraphRetriever, generator, *, max_agent_steps: int = 6):
        self.retriever = retriever
        self.generator = generator
        self.agent = (GraphAgent(retriever, generator, max_steps=max_agent_steps)
                      if hasattr(generator, "agent_step") else None)

    def ask(self, question: str, *, entity: str | None = None) -> dict:
        usage_start = len(getattr(self.generator, "token_usage", []))
        retrieved = self.retriever.retrieve(question, entity=entity)
        translated_query = None
        needs_translation = (not retrieved["facts"] or
                             (retrieved["ambiguous"] and re.search(r"[\u4e00-\u9fff]", question)))
        # Overview questions ("有哪些/多少种/列举...") have no entity to anchor on;
        # an English rewrite cannot help entity matching, so skip the call.
        overview_intent = bool(re.search(
            r"有哪些|多少种|列举|哪些类型|都有什么|有哪些类型|分类", question))
        if overview_intent and retrieved["mode"] == "entity_missing":
            needs_translation = False
        if needs_translation and entity is None and hasattr(self.generator, "translate_for_lookup"):
            translated_query = self.generator.translate_for_lookup(question)
            if translated_query and translated_query.casefold() != question.strip().casefold():
                translated = self.retriever.retrieve(
                    translated_query, entity=retrieved["anchor"] if retrieved["facts"] else None)
                if translated["facts"]:
                    translated["mode"] = ("translated_relation_filter" if retrieved["facts"]
                                          else "translated_exact_entity")
                    retrieved = translated
        if self.agent is not None:
            retrieved = self.agent.run(question, seed=retrieved)
        public = {"question": question, "retrieval": {
            "mode": retrieved["mode"], "anchor": retrieved["anchor"],
            "ambiguous": retrieved["ambiguous"],
            "candidate_documents": retrieved.get("candidate_documents", 0),
            "translated_query": translated_query,
            "agent_status": retrieved.get("agent_status"),
            "agent_assessment": retrieved.get("agent_assessment", {}),
            "agent_rounds": retrieved.get("agent_rounds", 0),
            "agent_trace": retrieved.get("agent_trace", []),
            "evidence_truncated": retrieved.get("evidence_truncated", False),
            "candidate_values": retrieved.get("candidate_values", 0),
        }, "evidence": {"documents": retrieved["documents"], "facts": retrieved["facts"]}}
        def attach_usage() -> None:
            usage, totals, context_limit = _usage_snapshot(self.generator, usage_start)
            public["retrieval"]["token_usage"] = usage
            public["retrieval"]["token_totals"] = totals
            public["retrieval"]["context_limit"] = context_limit

        attach_usage()
        public["evidence"]["catalogs"] = retrieved.get("catalogs", [])
        if not retrieved["facts"] and not retrieved.get("catalogs"):
            public["answer"] = "当前图谱及关联原文中没有找到足以回答该问题的证据。请提供更具体的工件或工序名称。"
            public["evidence_explanation"] = ""
            attach_usage()
            return public
        agent_status = retrieved.get("agent_status")
        if agent_status in {"insufficient", "step_limit"}:
            public["answer"] = ("当前代理尚未确认图谱证据已经完整，无法可靠完成该问题。"
                                "请缩小工件、工序或参数范围后重试。")
            public["evidence_explanation"] = ("已检索到的出处：" + "、".join(
                _document_citation(document) for document in retrieved["documents"])
                if retrieved["documents"] else evidence_explanation(retrieved, [
                    item["id"] for item in retrieved.get("catalogs", [])]))
            public["cited_evidence"] = ([document["id"] for document in retrieved["documents"]]
                                         + [item["id"] for item in retrieved.get("catalogs", [])])
            attach_usage()
            return public
        answer_mode = retrieved.get("answer_mode", "single")
        if retrieved["ambiguous"] and answer_mode == "single":
            public["answer"] = ("该名称对应多条加工记录，现有问题无法确定唯一工件及其参数。"
                                "请补充工件全称、工序或原文行号。")
            candidates = retrieved["documents"]
            public["evidence_explanation"] = "候选出处：" + "、".join(
                _document_citation(item) for item in candidates)
            attach_usage()
            return public
        if answer_mode in {"list", "aggregate", "compare"} and retrieved.get("catalogs"):
            # Catalogue answers are assembled by code, not generated: enumeration
            # of many values cannot fit a generation limit and free-form prose
            # reliably loses or misbinds citations (the empty-answer failure).
            answer, citations, explanation = compose_catalog_answer(retrieved)
            public["answer"] = answer
            public["evidence_explanation"] = explanation
            public["cited_evidence"] = citations
            attach_usage()
            return public
        answer_messages = build_messages(question, retrieved)
        if retrieved.get("catalogs"):
            # Catalogue answers need more room than a single-parameter answer,
            # while the prompt still asks the model to summarize by category.
            try:
                raw_answer = _call_generator(self.generator, answer_messages,
                                              max_new_tokens=512, stage="answer")
            except TypeError:
                raw_answer = _call_generator(self.generator, answer_messages, stage="answer")
        else:
            raw_answer = _call_generator(self.generator, answer_messages, stage="answer")
        valid = {item["id"] for item in retrieved["documents"]}
        valid.update(item["id"] for item in retrieved["facts"] if not item["type_warning"])
        valid.update(item["id"] for item in retrieved.get("catalogs", []))
        answer, citations = prose_answer(raw_answer, valid)
        if retrieved.get("catalogs"):
            answer, citations = _drop_mismatched_catalog_citations(
                answer, citations, retrieved["catalogs"])
        suspect = [fact for fact in retrieved["facts"] if fact["type_warning"]
                   and fact["object"].casefold() in answer.casefold()]
        if suspect:
            sources = list(dict.fromkeys(fact["document_id"] for fact in suspect))
            public["answer"] = ("关联图谱中的类别关系被误接到带单位的量值，生成回答无法通过证据校验。"
                                "现有记录不足以可靠确定所问类别的具体名称，请核对原文。")
            public["evidence_explanation"] = (
                "存在类型冲突的抽取项：" + "、".join(
                    f"[{fact['id']}] {fact['relation']}→{fact['object']}"
                    for fact in suspect) + "；对应原文 " + "、".join(
                    _document_citation(doc)
                    for doc in retrieved["documents"] if doc["id"] in sources)
            )
            public["cited_evidence"] = sources
            public["grounding_warning"] = "category_quantity_conflict"
            attach_usage()
            return public
        public["answer"] = answer
        public["evidence_explanation"] = evidence_explanation(retrieved, citations)
        public["cited_evidence"] = list(dict.fromkeys(citations))
        attach_usage()
        return public
