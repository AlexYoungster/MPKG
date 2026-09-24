"""CoT verifier that shares retrieval and type checks with the standard path."""

import copy
import re

from edc.schema_canonicalization import SchemaCanonicalizer
from edc.utils import llm_utils


class SchemaCanonicalizerCoT(SchemaCanonicalizer):
    def __init__(self, *args, max_tokens=256, **kwargs):
        super().__init__(*args, **kwargs)
        if max_tokens < 32:
            raise ValueError("CoT verification needs at least 32 output tokens")
        self.max_tokens = max_tokens
        self.verification_trace = []

    @staticmethod
    def extract_final_option(output, valid_letters):
        """Read the final decision, never an option mentioned in the analysis."""
        final = re.findall(
            r"(?:最终答案|Final\s+Answer|最后选择)\s*[:：]\s*(?:选项\s*)?([A-Z])\b",
            output,
            flags=re.IGNORECASE,
        )
        if final:
            option = final[-1].upper()
            return option if option in valid_letters else None
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
        candidates = list(candidate_relation_definition_dict)
        letters = [chr(ord("A") + index) for index in range(len(candidates))]
        none_letter = chr(ord("A") + len(candidates))
        query_family = self._relation_family(
            query_triplet[1], query_relation_definition, query_triplet[2]
        )
        query_definition = query_relation_definition
        if query_family:
            query_definition += f"；语义类别={self._family_label(query_family)}"

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
            raw = llm_utils.generate_completion_transformers(
                messages, self.verifier_model, self.verifier_tokenizer,
                answer_prepend="", max_new_token=self.max_tokens,
            )
        else:
            raw = llm_utils.openai_chat_completion(
                self.verifier_openai_model, None, messages, max_tokens=self.max_tokens
            )

        selected_option = self.extract_final_option(raw, set(letters + [none_letter]))
        selected_relation = (
            candidates[letters.index(selected_option)] if selected_option in letters else None
        )
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

        self.verification_trace.append({
            "query_triplet": query_triplet,
            "candidate_relations": candidates,
            "raw_output": raw,
            "selected_option": selected_option,
            "selected_relation": selected_relation,
            "family_override": family_override,
        })
        if selected_relation is None:
            return None
        canonical = copy.deepcopy(query_triplet)
        canonical[1] = selected_relation
        return canonical


# Keep the original exported class name for callers that import it directly.
SchemaCanonicalizer_CoT = SchemaCanonicalizerCoT
