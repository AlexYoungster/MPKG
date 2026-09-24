from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class SchemaDefiner:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def define_schema(
        self,
        input_text_str: str,
        extracted_triplets_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
    ) -> List[List[str]]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present
       
        relations_present = set()
        triples_by_relation = {}
        for t in extracted_triplets_list:
            relations_present.add(t[1])
            triples_by_relation.setdefault(t[1], []).append(t)

        filled_prompt = prompt_template_str.format_map(
            {
                "text": input_text_str,
                "few_shot_examples": few_shot_examples_str,
                "relations": sorted(relations_present),
                "triples": extracted_triplets_list,
            }
        )
        logger.info(f"填充后的完整提示前200个字符: {filled_prompt[:200]}...")
        messages = [{"role": "user", "content": filled_prompt}]
        completion = self._generate_definition(messages)
        logger.info(f"模型原始输出: {completion}")
        relation_definition_dict = self._custom_parse_relation_definition(completion, relations_present)
        invalid_relations = [
            relation
            for relation in sorted(relations_present)
            if not self._is_abstract_definition(
                relation, relation_definition_dict.get(relation, ""), triples_by_relation.get(relation, [])
            )
        ]

        if invalid_relations:
            repair_triples = [
                triple
                for relation in invalid_relations
                for triple in triples_by_relation.get(relation, [])
            ]
            repair_prompt = prompt_template_str.format_map(
                {
                    "text": input_text_str,
                    "few_shot_examples": "",
                    "relations": invalid_relations,
                    "triples": repair_triples,
                }
            )
            repair_prompt += (
                "\n\n前次输出未通过抽象性检查。请重新查看这些关系对应的全部实例，"
                "概括主体类别、客体类别及实例共有的关系含义。定义必须解释通用语义，"
                "不能复述关系名作为谓词含义，不能照抄任一实例的实体、数值、单位或情境。"
                "严格使用规定字段，每个关系一行，只输出这些关系的修订定义。"
            )
            repair_completion = self._generate_definition(
                [{"role": "user", "content": repair_prompt}]
            )
            repaired_definitions = self._custom_parse_relation_definition(
                repair_completion, set(invalid_relations)
            )
            for relation in invalid_relations:
                repaired = repaired_definitions.get(relation, "")
                if self._is_abstract_definition(
                    relation, repaired, triples_by_relation.get(relation, [])
                ):
                    relation_definition_dict[relation] = repaired

        for relation in sorted(relations_present):
            if not self._is_abstract_definition(
                relation, relation_definition_dict.get(relation, ""), triples_by_relation.get(relation, [])
            ):
                relation_definition_dict[relation] = self._generic_instance_definition(
                    relation, triples_by_relation.get(relation, [])
                )

        logger.info(f"结构化关系定义解析结果: {relation_definition_dict}")
        missing_relations = [rel for rel in relations_present if rel not in relation_definition_dict]
        if len(missing_relations) != 0:
            logger.debug(f"Relations {missing_relations} received generic fallback definitions")
        return relation_definition_dict

    def _generate_definition(self, messages):
        if self.openai_model is None:
            return llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, max_new_token=768
            )
        return llm_utils.openai_chat_completion(self.openai_model, None, messages)

    @staticmethod
    def _is_abstract_definition(relation, definition, relation_triples):
        required_fields = ("主体类别=", "方向=", "客体类别=", "谓词语义=", "范围边界=")
        if not definition or any(field not in definition for field in required_fields):
            return False
        if "方向=主体→客体" not in definition:
            return False
        if re.search(r"\d|mm|rpm|r/min|ra\d", definition, re.IGNORECASE):
            return False

        field_values = {}
        for field in ("主体类别", "客体类别", "谓词语义", "范围边界"):
            match = re.search(rf"{field}=([^；;]+)", definition)
            if not match or not match.group(1).strip():
                return False
            field_values[field] = match.group(1).strip()
        predicate = field_values["谓词语义"]
        if predicate == relation or predicate.startswith("按关系名"):
            return False

        for triple in relation_triples:
            for instance in (triple[0], triple[2]):
                if instance and len(instance.strip()) > 1 and instance.strip() in definition:
                    return False
        return True

    @staticmethod
    def _generic_instance_definition(relation, relation_triples):
        """Abstract an invalid SD result from relation and repeated instance cues."""
        instances = " ".join(
            f"{triple[0]} {triple[2]}" for triple in relation_triples if len(triple) == 3
        ).lower()
        text = f"{relation} {instances}".lower()
        relation_lower = relation.lower()

        # English labels need explicit word-level cues. In particular, testing
        # for the substring "ra" in all instances falsely classifies words
        # such as "operation" as a surface-roughness relation.
        if any(term in relation_lower for term in ("surface roughness", "roughness")):
            subject, object_, predicate, boundary = (
                "workpiece or processed surface", "surface roughness parameter", "states the required or measured surface roughness", "surface quality requirement or result"
            )
        elif "cutting speed" in relation_lower:
            subject, object_, predicate, boundary = (
                "machining operation", "cutting velocity parameter", "states the cutting edge speed relative to the workpiece surface", "primary cutting motion"
            )
        elif "spindle speed" in relation_lower:
            subject, object_, predicate, boundary = (
                "machining operation", "spindle rotation parameter", "states the rotational speed of the machine spindle", "spindle motion"
            )
        elif "feed rate" in relation_lower or "feed per revolution" in relation_lower:
            subject, object_, predicate, boundary = (
                "machining operation", "feed distance parameter", "states tool displacement relative to the workpiece per revolution or stroke", "feed motion"
            )
        elif "feed speed" in relation_lower or "table speed" in relation_lower:
            subject, object_, predicate, boundary = (
                "machining operation", "linear feed speed parameter", "states displacement of a feed component per unit time", "feed motion"
            )
        elif any(term in relation_lower for term in ("depth of cut", "cutting depth", "infeed depth")):
            subject, object_, predicate, boundary = (
                "machining operation", "cutting depth parameter", "states the depth of material removed or entered during each pass", "cutting or grinding depth"
            )
        elif any(term in relation_lower for term in ("geometric tolerance", "parallelism", "perpendicularity")):
            subject, object_, predicate, boundary = (
                "workpiece or geometric feature", "geometric tolerance parameter", "states the allowed deviation in shape orientation or position", "geometric accuracy requirement"
            )
        elif "tool material" in relation_lower:
            subject, object_, predicate, boundary = (
                "machining tool", "material type", "states the material from which the subject tool is made", "tool material property"
            )
        elif relation_lower in ("tool", "cutting tool", "grinding wheel", "abrasive"):
            subject, object_, predicate, boundary = (
                "machining operation", "tool or abrasive medium", "states that the operation uses the object as a working tool or medium", "tool use"
            )
        elif relation_lower in ("coolant", "cooling method"):
            subject, object_, predicate, boundary = (
                "machining operation", "cooling medium or method", "states the cooling means used by the operation", "process cooling"
            )
        elif "material" in relation_lower:
            subject, object_, predicate, boundary = (
                "workpiece or part", "material type", "states the material from which the subject is made", "workpiece material property"
            )
        elif "operation" in relation_lower or "process sequence" in relation_lower:
            subject, object_, predicate, boundary = (
                "workpiece or process", "manufacturing operation or step", "states the operation or sequence performed on the subject", "manufacturing process"
            )

        elif any(term in text for term in ("材料", "材质")):
            subject, object_, predicate, boundary = (
                "加工对象", "材料类别", "说明主体与其构成材料之间的组成关系", "主体的材料属性"
            )
        elif any(term in text for term in ("平行度", "垂直度", "位置度", "几何公差")):
            subject, object_, predicate, boundary = (
                "零件或加工对象", "几何精度指标", "表示主体需要满足的形状、方向或位置精度约束", "几何公差要求"
            )
        elif any(term in text for term in ("粗糙度", "表面质量")) or re.search(r"\bra\s*\d", text):
            subject, object_, predicate, boundary = (
                "零件或加工对象", "表面质量指标", "表示主体表面需要满足的粗糙程度要求", "表面质量要求"
            )
        elif any(term in text for term in ("rpm", "r/min", "主轴转速", "转/分")):
            subject, object_, predicate, boundary = (
                "加工过程", "旋转速度参数", "表示加工过程中主轴的旋转速率", "主轴运动参数"
            )
        elif any(term in text for term in ("mm/r", "mm/转", "进给率", "进给量")):
            subject, object_, predicate, boundary = (
                "加工过程", "每转进给参数", "表示刀具相对于工件的每转进给距离", "进给运动参数"
            )
        elif any(term in text for term in ("背吃刀", "进给深度", "切削深度", "深度")):
            subject, object_, predicate, boundary = (
                "加工过程", "深度参数", "表示加工过程中每次切入或进给所对应的切削深度", "加工深度参数"
            )
        elif any(term in text for term in ("工作台速度", "进给速度", "进给运动部件")):
            subject, object_, predicate, boundary = (
                "加工过程", "进给运动速度参数", "表示进给运动部件相对于工件的线位移速率", "进给运动参数"
            )
        elif any(term in text for term in ("砂轮", "刀具", "切削工具", "磨具", "工具")) and any(
            term in relation for term in ("使用", "采用", "配备", "装备")
        ):
            subject, object_, predicate, boundary = (
                "加工过程", "加工工具", "表示加工过程与其所采用工具之间的使用关系", "工具使用关系"
            )
        elif "冷却" in text:
            subject, object_, predicate, boundary = (
                "加工过程", "冷却介质或方式", "表示加工过程采用的冷却手段", "加工冷却关系"
            )
        elif any(term in text for term in ("加工工序", "加工方法", "加工过程")):
            subject, object_, predicate, boundary = (
                "加工对象", "工艺过程或步骤", "表示加工对象与其执行工艺之间的关系", "制造加工过程"
            )
        else:
            subject, object_, predicate, boundary = (
                "关系实例中的主体类别", "关系实例中的客体类别",
                f"概括关系名“{relation}”在实例中表示的共同语义", "以同一关系的实例共性为准"
            )

        return (
            f"主体类别={subject}；方向=主体→客体；客体类别={object_}；"
            f"谓词语义={predicate}；范围边界={boundary}。"
        )
    
    def _custom_parse_relation_definition(self, raw_text, relations_present):
        """Parse only requested relation labels and tolerate list/colon variants."""
        result = {}

        ordered_relations = sorted(relations_present, key=len, reverse=True)
        for line in raw_text.splitlines():
            cleaned_line = re.sub(r"^\s*(?:[-*•]\s*|\d+[.、)]\s*)", "", line).strip()
            for relation in ordered_relations:
                match = re.match(
                    rf"^{re.escape(relation)}\s*[：:]\s*(.+?)\s*$", cleaned_line
                )
                if match:
                    result[relation] = match.group(1).strip()
                    break

        return result
