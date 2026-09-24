"""CoT verifier that shares retrieval and type checks with the standard path.

CoT（Chain-of-Thought）验证器模块。

在 Schema Canonicalization（关系标准化）阶段，对开放三元组 (头实体, 开放关系, 尾实体)
决定其开放关系应映射到目标模式中的哪个规范关系。与父类 SchemaCanonicalizer 的
"短答案验证"（max_new_token=5，模型直接输出一个选项字母）不同，本模块让验证 LLM
先输出一段比较推理文本、再给出"最终答案：X"，并从推理文本中可靠地解析出最终选项。

与父类共享的基础设施（不在此重复实现，直接继承复用）：
- 候选关系召回：retrieve_similar_relations()（多路检索 + RRF 融合）；
- 语义类别（family）判定：_relation_family() / _family_label()；
- 候选召回结果由父类 canonicalize() 传入 llm_verify()。
"""

import copy
import re

from edc.schema_canonicalization import SchemaCanonicalizer
from edc.utils import llm_utils


class SchemaCanonicalizerCoT(SchemaCanonicalizer):
    def __init__(self, *args, max_tokens=256, **kwargs):
        super().__init__(*args, **kwargs)
        # CoT 需要同时容纳"推理过程 + 最终答案"，过小的预算会让推理被截断，
        # 导致 extract_final_option 拿不到"最终答案：X"而验证失败，故设下限 32。
        if max_tokens < 32:
            raise ValueError("CoT verification needs at least 32 output tokens")
        self.max_tokens = max_tokens
        # 每条三元组一次验证的轨迹（原始输出/所选选项/是否触发类别兜底），
        # 由 edc_framework 按 entry 切片收集后写入 result_at_each_stage.json。
        self.verification_trace = []

    @staticmethod
    def extract_final_option(output, valid_letters):
        """Read the final decision, never an option mentioned in the analysis."""
        # 解析优先级 1：显式结论标记。只认 "最终答案/Final Answer/最后选择：X" 的
        # 最后一次出现（[-1]），防止推理前半段提到的过渡选项被误当作结论。
        final = re.findall(
            r"(?:最终答案|Final\s+Answer|最后选择)\s*[:：]\s*(?:选项\s*)?([A-Z])\b",
            output,
            flags=re.IGNORECASE,
        )
        if final:
            option = final[-1].upper()
            return option if option in valid_letters else None
        # 解析优先级 2：无显式标记时，退化到"最后一行即答案"的格式约定；
        # 同样校验字母必须在合法候选集合内，否则视为弃权（返回 None）。
        lines = [line.strip().strip("*` ") for line in output.splitlines() if line.strip()]
        if lines:
            match = re.fullmatch(r"(?:选项\s*)?([A-Z])[.。]?", lines[-1], re.IGNORECASE)
            if match:
                option = match.group(1).upper()
                return option if option in valid_letters else None
        return None

    def llm_verify(
        self,
        input_text_str,
        query_triplet,
        query_relation_definition,
        prompt_template_str,
        candidate_relation_definition_dict,
        relation_example_dict=None,
    ):
        """构建带选项的多选题提示，让模型先推理再作答; `max_tokens` 默认 256, 区别于标准路径只让模型答 1 个字母"""
        # 候选选项字母 A..N 按召回顺序分配；none_letter 紧随其后，表示"都不匹配"。
        candidates = list(candidate_relation_definition_dict)
        letters = [chr(ord("A") + index) for index in range(len(candidates))]
        none_letter = chr(ord("A") + len(candidates))
        # 判定查询三元组的语义类别（如 主轴转速/每转进给/材料 等），
        # 并把类别标签追加进查询关系定义，给 LLM 一个粗粒度的类型约束信号。
        query_family = self._relation_family(
            query_triplet[1], query_relation_definition, query_triplet[2]
        )
        query_definition = query_relation_definition
        if query_family:
            query_definition += f"；语义类别={self._family_label(query_family)}"

        # 组装选择题：每个候选关系附定义与类别标签，再追加一个 None 选项。
        choices = []
        for letter, relation in zip(letters, candidates):
            description = candidate_relation_definition_dict[relation]
            family = self._relation_family(relation, description)
            if family:
                description += f"【语义类别={self._family_label(family)}】"
            choices.append(f"{letter}. '{relation}': {description}")
        choices.append(f"{none_letter}. None of the above.")
        prompt = prompt_template_str.format_map({
            "input_text": input_text_str,
            "query_triplet": query_triplet,
            "query_relation": query_triplet[1],
            "query_relation_definition": query_definition,
            "choices": "\n".join(choices),
        })
        messages = [{"role": "user", "content": prompt}]
        if self.verifier_openai_model is None:
            # 本地 HF 模型：answer_prepend 为空，允许模型先自由推理再作答；
            # max_new_token 用 CoT 预算（父类短答案版只有 5）。
            raw = llm_utils.generate_completion_transformers(
                messages, self.verifier_model, self.verifier_tokenizer,
                answer_prepend="", max_new_token=self.max_tokens,
            )
        else:
            raw = llm_utils.openai_chat_completion(
                self.verifier_openai_model, None, messages, max_tokens=self.max_tokens
            )

        # 只从"最终答案"处解析选项，映射回候选关系；选 None 或解析失败则为 None。
        selected_option = self.extract_final_option(raw, set(letters + [none_letter]))
        selected_relation = (
            candidates[letters.index(selected_option)] if selected_option in letters else None
        )
        # 语义类别兜底（family_override）：与父类短答案版相同的守卫——
        # 当候选中"语义类别与查询一致"的关系唯一，而 LLM 的选择类别不符（或弃权）时，
        # 无视 LLM 选择、强制改判为该唯一兼容候选。这可在 CoT 推理跑偏时兜住类型错误。
        family_override = False
        if query_family:
            compatible = [
                relation for relation in candidates
                if self._relation_family(relation, self.schema_dict[relation]) == query_family
            ]
            selected_family = (
                self._relation_family(selected_relation, self.schema_dict[selected_relation])
                if selected_relation in self.schema_dict else None
            )
            if len(compatible) == 1 and selected_family != query_family:
                selected_relation = compatible[0]
                family_override = True

        # 记录本条验证的完整轨迹，供结果落盘（canonicalization_reasoning）与评测对比。
        self.verification_trace.append({
            "query_triplet": query_triplet,
            "candidate_relations": candidates,
            "raw_output": raw,
            "selected_option": selected_option,
            "selected_relation": selected_relation,
            "family_override": family_override,
        })
        # 未选中任何候选 → 放弃标准化（上层 canonicalize 会按 enrich 决定是否补入模式）。
        if selected_relation is None:
            return None
        # 深拷贝原三元组，仅替换中段关系为规范关系，头/尾实体保持不变。
        # 注意不可原地修改 query_triplet（评测脚本用它回溯 OIE 原始输出）。
        canonical = copy.deepcopy(query_triplet)
        canonical[1] = selected_relation
        return canonical


# Keep the original exported class name for callers that import it directly.
SchemaCanonicalizer_CoT = SchemaCanonicalizerCoT
