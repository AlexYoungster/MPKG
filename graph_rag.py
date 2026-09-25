"""Minimal source-grounded Graph RAG over the SQLite graph built by MPKG."""

import os
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
    def __init__(self, database: Path, *, max_documents=3):
        self.database = Path(database).resolve()
        with closing(open_graph(self.database)) as connection:
            connection.execute("SELECT 1 FROM occurrences LIMIT 1").fetchone()
        self.max_documents = max_documents

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
        import json
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
在结论句末就地标注证据编号，如 [E2] 或 [D1]，再用一句话说明工件与工序的关系和出处。不要另附三元组清单、表格或未被询问的工艺建议。
不要把其他文本中同名工序的参数移到当前工件上。
若证据不足、主体不明确或无法区分多个工件，明确说无法确定，并说明缺少什么。不要编造工艺参数或建议。"""


def build_messages(question: str, retrieved: dict) -> list[dict]:
    lines = [f"问题：{question}", "", "以下证据按原始文本分组，同组图谱事实才属于同一次记录："]
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
        lines.append("该实体出现在更多原文中；当前仅列出部分记录。不能据此给出唯一参数时请要求补充工件或来源行号。")
    lines.append("请只回答问题所问，不要复述所有证据。")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


class QwenGenerator:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", *, cache_root: Path | None = None,
                 offline=False, max_new_tokens=320):
        self.model_name = model_name
        self.cache_root = cache_root or Path(os.environ.get(
            "MPKG_MODEL_CACHE", Path(__file__).resolve().parent / ".cache" / "models"))
        self.offline = offline
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

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

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        self._load()
        import torch
        from transformers import GenerationConfig

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        if inputs["input_ids"].shape[-1] > 4096:
            raise ValueError("检索上下文超过 4096 tokens；请提供更具体的工件名称")
        generation = GenerationConfig(
            do_sample=False, max_new_tokens=max_new_tokens or self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.model.generation_config.eos_token_id,
        )
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=generation)
        new_ids = output[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        answer = re.sub(r"^<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
        return answer

    def translate_for_lookup(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "把用户的机械加工问题改写成一句简短的英文检索句。"
             "准确翻译工件和工序名称，保留型号、数值、单位；不要回答问题，不要增添事实。只输出英文句子。"},
            {"role": "user", "content": question},
        ]
        output = self.generate(messages, max_new_tokens=80)
        return output.splitlines()[0].strip().strip('"“”') if output else ""


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
    if not statements and documents:
        first = next(iter(documents.values()))
        statements.append(f"[{first['id']}] 对应原文 {_document_citation(first)}")
    return " ".join(statements)


def prose_answer(raw: str, valid_citations: set[str]) -> tuple[str, list[str]]:
    """Keep the answer paragraph and move redundant triple dumps to evidence JSON."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    answer = paragraphs[0] if paragraphs else ""
    answer = re.sub(r"\[([DE]\d+)\]",
                    lambda match: match.group(0) if match.group(1) in valid_citations else "",
                    answer)
    cited = [key for key in dict.fromkeys(re.findall(r"\[([DE]\d+)\]", answer))
             if key in valid_citations]
    answer = re.sub(
        r"\[E\d+\]\s*[^。；;\n]*?--[^。；;\n]*?-->[^。；;\n]*(?:[。；;]|$)",
        "", answer,
    )
    answer = re.sub(r"\s+", " ", answer).strip()
    retained = set(re.findall(r"\[([DE]\d+)\]", answer))
    missing = [key for key in cited if key not in retained]
    if missing:
        answer += " 依据图谱与原文记录 " + "".join(f"[{key}]" for key in missing) + "。"
    return answer, cited


class GraphRAG:
    def __init__(self, retriever: GraphRetriever, generator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str, *, entity: str | None = None) -> dict:
        retrieved = self.retriever.retrieve(question, entity=entity)
        translated_query = None
        needs_translation = (not retrieved["facts"] or
                             (retrieved["ambiguous"] and re.search(r"[\u4e00-\u9fff]", question)))
        if needs_translation and entity is None and hasattr(self.generator, "translate_for_lookup"):
            translated_query = self.generator.translate_for_lookup(question)
            if translated_query and translated_query.casefold() != question.strip().casefold():
                translated = self.retriever.retrieve(
                    translated_query, entity=retrieved["anchor"] if retrieved["facts"] else None)
                if translated["facts"]:
                    translated["mode"] = ("translated_relation_filter" if retrieved["facts"]
                                          else "translated_exact_entity")
                    retrieved = translated
        public = {"question": question, "retrieval": {
            "mode": retrieved["mode"], "anchor": retrieved["anchor"],
            "ambiguous": retrieved["ambiguous"],
            "candidate_documents": retrieved.get("candidate_documents", 0),
            "translated_query": translated_query,
        }, "evidence": {"documents": retrieved["documents"], "facts": retrieved["facts"]}}
        if not retrieved["facts"]:
            public["answer"] = "当前图谱及关联原文中没有找到足以回答该问题的证据。请提供更具体的工件或工序名称。"
            public["evidence_explanation"] = ""
            return public
        if retrieved["ambiguous"]:
            public["answer"] = ("该名称对应多条加工记录，现有问题无法确定唯一工件及其参数。"
                                "请补充工件全称、工序或原文行号。")
            candidates = retrieved["documents"]
            public["evidence_explanation"] = "候选出处：" + "、".join(
                _document_citation(item) for item in candidates)
            return public
        raw_answer = self.generator.generate(build_messages(question, retrieved))
        valid = {item["id"] for item in retrieved["documents"]}
        valid.update(item["id"] for item in retrieved["facts"] if not item["type_warning"])
        answer, citations = prose_answer(raw_answer, valid)
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
            return public
        public["answer"] = answer
        public["evidence_explanation"] = evidence_explanation(retrieved, citations)
        public["cited_evidence"] = list(dict.fromkeys(citations))
        return public
