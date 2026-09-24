from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


class Extractor:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint: str = None,
        relations_hint: str = None,
    ) -> List[List[str]]:
        assert (entities_hint is None and relations_hint is None) or (
            relations_hint is not None and relations_hint is not None
        )

        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )

        messages = [{"role": "user", "content": filled_prompt}]

        if self.openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            completion = llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, answer_prepend="Triplets: "
            )
        else:
            completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
        extracted_triplets_list = llm_utils.parse_raw_triplets(completion)

        # Ask a separate pass for explicitly stated facts omitted by the first
        # extraction. Keep the first pass intact and add only new triples.
        audit_prompt = (
            "请核对原文与已抽取的三元组，找出已抽取列表遗漏的、且原文明确陈述的事实。"
            "将每项遗漏事实写成[主体, 关系, 客体]。保持原文的实体边界、关系方向、限定信息、数值和单位。"
            "只补充遗漏事实，不重复已有三元组，不根据常识推断，不输出解释。"
            "没有遗漏时只输出[]。\n\n"
            f"原文：{input_text_str}\n"
            f"已抽取三元组：{extracted_triplets_list}"
        )
        audit_messages = [{"role": "user", "content": audit_prompt}]
        if self.openai_model is None:
            audit_completion = llm_utils.generate_completion_transformers(
                audit_messages, self.model, self.tokenizer, max_new_token=256, answer_prepend="Triplets: "
            )
        else:
            audit_completion = llm_utils.openai_chat_completion(self.openai_model, None, audit_messages)
        omitted_triplets = llm_utils.parse_raw_triplets(audit_completion)

        seen_triplets = {tuple(triplet) for triplet in extracted_triplets_list}
        for triplet in omitted_triplets:
            normalized_triplet = tuple(triplet)
            if normalized_triplet not in seen_triplets:
                extracted_triplets_list.append(triplet)
                seen_triplets.add(normalized_triplet)
        return extracted_triplets_list
