from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class SchemaCanonicalizer:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(
        self,
        target_schema_dict: dict,
        embedder: SentenceTransformer,
        verify_model: AutoTokenizer = None,
        verify_tokenizer: AutoTokenizer = None,
        verify_openai_model: AutoTokenizer = None,
    ) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        assert verify_openai_model is not None or (verify_model is not None and verify_tokenizer is not None)
        self.verifier_model = verify_model
        self.verifier_tokenizer = verify_tokenizer
        self.verifier_openai_model = verify_openai_model
        self.schema_dict = target_schema_dict

        self.embedder = embedder

        self._rebuild_schema_index()

    @staticmethod
    def _schema_document(relation: str, definition: str) -> str:
        return f"关系名称：{relation}\n关系语义：{definition}"

    def _rebuild_schema_index(self):
        """Index relation names and definitions for dense and lexical retrieval."""
        self.schema_embedding_dict = {}
        self.schema_relation_list = list(self.schema_dict.keys())
        self.schema_definition_list = [self.schema_dict[relation] for relation in self.schema_relation_list]
        self.schema_document_list = [
            self._schema_document(relation, self.schema_dict[relation])
            for relation in self.schema_relation_list
        ]

        print("Embedding target schema...")
        for relation, document in tqdm(zip(self.schema_relation_list, self.schema_document_list),
                                       total=len(self.schema_relation_list)):
            self.schema_embedding_dict[relation] = self.embedder.encode(document)

        self.schema_embedding_matrix = np.asarray(
            [self.schema_embedding_dict[relation] for relation in self.schema_relation_list],
            dtype=np.float32,
        )
        self.schema_name_embedding_matrix = np.asarray(
            self.embedder.encode(self.schema_relation_list), dtype=np.float32
        )
        self.schema_definition_embedding_matrix = np.asarray(
            self.embedder.encode(self.schema_definition_list), dtype=np.float32
        )
        if len(self.schema_embedding_matrix):
            norms = np.linalg.norm(self.schema_embedding_matrix, axis=1, keepdims=True)
            self.schema_embedding_matrix = self.schema_embedding_matrix / np.maximum(norms, 1e-12)
            name_norms = np.linalg.norm(self.schema_name_embedding_matrix, axis=1, keepdims=True)
            self.schema_name_embedding_matrix /= np.maximum(name_norms, 1e-12)
            definition_norms = np.linalg.norm(self.schema_definition_embedding_matrix, axis=1, keepdims=True)
            self.schema_definition_embedding_matrix /= np.maximum(definition_norms, 1e-12)

        # Character n-grams work well for Chinese relation labels without requiring
        # a tokenizer or an additional model.
        self.document_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 3), sublinear_tf=True, norm="l2"
        )
        self.document_tfidf = self.document_vectorizer.fit_transform(self.schema_document_list)
        self.name_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), sublinear_tf=True, norm="l2"
        )
        self.name_tfidf = self.name_vectorizer.fit_transform(self.schema_relation_list)

    @staticmethod
    def _relation_family(relation: str, definition: str = "", value: str = ""):
        """Infer a broad machining relation family for compatibility checks."""
        text = f"{relation} {definition} {value}".lower()
        if any(term in text for term in ("ra", "粗糙度", "表面质量")):
            return "surface_roughness"
        if any(term in text for term in ("主轴转速", "rpm", "r/min", "转/分", "每分钟转数")):
            return "spindle_speed"
        if any(term in text for term in ("mm/r", "mm/转", "每转进给", "进给量", "进给率")):
            return "feed_per_revolution"
        if any(term in text for term in ("渗碳层深度", "渗氮层深度", "硬化层深度")):
            return "heat_treatment_layer_depth"
        if any(term in text for term in ("背吃刀", "切削深度", "进给深度", "切入深度", "深度")):
            return "cut_depth"
        if any(term in text for term in ("工作台速度", "进给速度", "进给运动速度", "进给运动部件")):
            return "feed_motion_speed"
        if any(term in text for term in ("切削速度", "切削线速度", "切削刃", "刀具相对于工件的线速度")):
            return "cutting_speed"
        if any(term in text for term in ("平行度", "垂直度", "位置度", "几何公差")):
            return "geometric_tolerance"
        if any(term in relation.lower() for term in ("材料", "材质", "material")):
            return "material"
        if "使用设备" in text or "加工设备使用" in text:
            return "equipment_use"
        if relation.startswith(("使用", "采用", "配备", "装备")) and any(
            term in text for term in ("砂轮", "刀具", "切削工具", "加工工具", "磨具")
        ):
            return "tool_use"
        if "几何要素" in text or "几何精度" in text:
            return "geometric_tolerance"
        return None

    @staticmethod
    def _family_label(family: str) -> str:
        labels = {
            "surface_roughness": "表面粗糙度参数",
            "spindle_speed": "主轴旋转参数",
            "feed_per_revolution": "每转进给参数",
            "cut_depth": "切削深度参数",
            "heat_treatment_layer_depth": "热处理层深度参数",
            "feed_motion_speed": "进给运动速度参数",
            "cutting_speed": "切削线速度参数",
            "geometric_tolerance": "几何公差要求",
            "material": "材料关系",
            "tool_use": "加工工具使用关系",
            "equipment_use": "加工设备使用关系",
        }
        return labels.get(family, family)

    def retrieve_similar_relations(
        self,
        query_relation_definition: str,
        top_k=5,
        query_relation: str = "",
        query_triplet: List[str] = None,
    ):
        target_relation_list = self.schema_relation_list
        if not target_relation_list:
            return {}, []

        query_document = self._schema_document(query_relation, query_relation_definition)
        if query_triplet and len(query_triplet) == 3:
            query_document += (
                f"\n主体实体：{query_triplet[0]}"
                f"\n客体实体：{query_triplet[2]}"
            )
        if "sts_query" in self.embedder.prompts:
            query_embedding = self.embedder.encode(query_document, prompt_name="sts_query")
        else:
            query_embedding = self.embedder.encode(query_document)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        query_embedding = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
        dense_scores = self.schema_embedding_matrix @ query_embedding

        if "sts_query" in self.embedder.prompts:
            name_query_embedding = self.embedder.encode(query_relation, prompt_name="sts_query")
            definition_query_embedding = self.embedder.encode(
                query_relation_definition, prompt_name="sts_query"
            )
        else:
            name_query_embedding = self.embedder.encode(query_relation)
            definition_query_embedding = self.embedder.encode(query_relation_definition)
        name_query_embedding = np.asarray(name_query_embedding, dtype=np.float32)
        definition_query_embedding = np.asarray(definition_query_embedding, dtype=np.float32)
        name_query_embedding /= max(float(np.linalg.norm(name_query_embedding)), 1e-12)
        definition_query_embedding /= max(float(np.linalg.norm(definition_query_embedding)), 1e-12)
        name_dense_scores = self.schema_name_embedding_matrix @ name_query_embedding
        definition_dense_scores = self.schema_definition_embedding_matrix @ definition_query_embedding

        document_query_tfidf = self.document_vectorizer.transform([query_document])
        lexical_scores = (document_query_tfidf @ self.document_tfidf.T).toarray()[0]
        name_query_tfidf = self.name_vectorizer.transform([query_relation or query_relation_definition])
        name_scores = (name_query_tfidf @ self.name_tfidf.T).toarray()[0]

        # Fuse independent views so a weak or overly specific SD definition
        # cannot suppress a strong relation-name or definition-only match.
        def rank_scores(scores):
            ranks = np.empty(len(target_relation_list), dtype=np.int32)
            ranks[np.argsort(-scores)] = np.arange(1, len(target_relation_list) + 1)
            return ranks

        dense_ranks = rank_scores(dense_scores)
        name_dense_ranks = rank_scores(name_dense_scores)
        definition_dense_ranks = rank_scores(definition_dense_scores)
        lexical_ranks = rank_scores(lexical_scores)
        name_ranks = rank_scores(name_scores)
        rrf_k = 60.0
        fused_scores = (
            0.75 / (rrf_k + dense_ranks)
            + 1.0 / (rrf_k + definition_dense_ranks)
            + 1.25 / (rrf_k + name_dense_ranks)
            + 0.75 / (rrf_k + lexical_ranks)
            + 1.25 / (rrf_k + name_ranks)
        )
        max_fused_score = (0.75 + 1.0 + 1.25 + 0.75 + 1.25) / (rrf_k + 1.0)
        fused_scores /= max_fused_score
        query_family = self._relation_family(
            query_relation,
            query_relation_definition,
            query_triplet[2] if query_triplet and len(query_triplet) == 3 else "",
        )
        if query_family:
            for idx, relation in enumerate(target_relation_list):
                candidate_family = self._relation_family(relation, self.schema_dict[relation])
                if candidate_family == query_family:
                    fused_scores[idx] += 0.18
                elif candidate_family:
                    fused_scores[idx] -= 0.12
        highest_score_indices = np.argsort(-fused_scores)[:top_k]

        return {
            target_relation_list[idx]: self.schema_dict[target_relation_list[idx]]
            for idx in highest_score_indices[:top_k]
        }, [float(fused_scores[idx]) for idx in highest_score_indices[:top_k]]

    def llm_verify(
        self,
        input_text_str: str,
        query_triplet: List[str],
        query_relation_definition: str,
        prompt_template_str: str,
        candidate_relation_definition_dict: dict,
        relation_example_dict: dict = None,
    ):
        canonicalized_triplet = copy.deepcopy(query_triplet)
        choice_letters_list = []
        choices = ""
        candidate_relations = list(candidate_relation_definition_dict.keys())
        candidate_relation_descriptions = []
        query_family = self._relation_family(
            query_triplet[1], query_relation_definition, query_triplet[2]
        )
        for relation, description in candidate_relation_definition_dict.items():
            family = self._relation_family(relation, description)
            if family:
                description += f"【语义类别={self._family_label(family)}】"
            candidate_relation_descriptions.append(description)
        query_definition_for_verification = query_relation_definition
        if query_family:
            query_definition_for_verification += (
                f"；语义类别={self._family_label(query_family)}"
            )
        for idx, rel in enumerate(candidate_relations):
            choice_letter = chr(ord("@") + idx + 1)
            choice_letters_list.append(choice_letter)
            choices += f"{choice_letter}. '{rel}': {candidate_relation_descriptions[idx]}\n"
            if relation_example_dict is not None:
                choices += f"Example: '{relation_example_dict[candidate_relations[idx]]['triple']}' can be extracted from '{candidate_relations[idx]['sentence']}'\n"
        choices += f"{chr(ord('@')+idx+2)}. None of the above.\n"

        verification_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text_str,
                "query_triplet": query_triplet,
                "query_relation": query_triplet[1],
                "query_relation_definition": query_definition_for_verification,
                "choices": choices,
            }
        )

        messages = [{"role": "user", "content": verification_prompt}]
        if self.verifier_openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            verification_result = llm_utils.generate_completion_transformers(
                messages, self.verifier_model, self.verifier_tokenizer, answer_prepend="Answer: ", max_new_token=5
            )
        else:
            verification_result = llm_utils.openai_chat_completion(
                self.verifier_openai_model, None, messages, max_tokens=1
            )

        selected_relation = None
        if verification_result and verification_result[0] in choice_letters_list:
            selected_relation = candidate_relations[choice_letters_list.index(verification_result[0])]

        if query_family:
            compatible_relations = [
                relation
                for relation, description in candidate_relation_definition_dict.items()
                if self._relation_family(relation, description) == query_family
            ]
            selected_family = (
                self._relation_family(selected_relation, self.schema_dict[selected_relation])
                if selected_relation in self.schema_dict
                else None
            )
            # A unique type-compatible candidate can safely recover from a
            # small verifier selecting an unrelated relation or abstaining.
            if len(compatible_relations) == 1 and selected_family != query_family:
                selected_relation = compatible_relations[0]

        if selected_relation is None:
            return None
        canonicalized_triplet[1] = selected_relation

        return canonicalized_triplet

    def canonicalize(
        self,
        input_text_str: str,
        open_triplet,
        open_relation_definition_dict: dict,
        verify_prompt_template: str,
        enrich=False,
    ):

        open_relation = open_triplet[1]

        if open_relation in self.schema_dict:
            # The relation is already canonical
            # candidate_relations, candidate_scores = self.retrieve_similar_relations(
            #     open_relation_definition_dict[open_relation]
            # )
            return open_triplet, {}

        candidate_relations = []
        candidate_scores = []

        if len(self.schema_dict) != 0:
            if open_relation not in open_relation_definition_dict:
                canonicalized_triplet = None
            else:
                candidate_relations, candidate_scores = self.retrieve_similar_relations(
                    open_relation_definition_dict[open_relation],
                    query_relation=open_relation,
                    query_triplet=open_triplet,
                )
                canonicalized_triplet = self.llm_verify(
                    input_text_str,
                    open_triplet,
                    open_relation_definition_dict[open_relation],
                    verify_prompt_template,
                    candidate_relations,
                    None,
                )
        else:
            canonicalized_triplet = None

        if canonicalized_triplet is None:
            # Cannot be canonicalized
            if enrich:
                self.schema_dict[open_relation] = open_relation_definition_dict[open_relation]
                self._rebuild_schema_index()
                canonicalized_triplet = open_triplet
        return canonicalized_triplet, dict(zip(candidate_relations, candidate_scores))
