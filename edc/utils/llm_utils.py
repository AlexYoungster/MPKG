import os
import re
import openai
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ast
from sentence_transformers import SentenceTransformer
from typing import List
import gc
import torch
import logging

logger = logging.getLogger(__name__)


def free_model(model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None):
    try:
        model.cpu()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(e)


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_sts(model: SentenceTransformer, text: str, prompt_name=None, prompt=None):
    embedding = model.encode(text, prompt_name=prompt_name, prompt=prompt)
    return embedding


def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    left_bracket_idx = raw_entities.index("[")
    right_bracket_idx = raw_entities.index("]")
    try:
        parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])
    except Exception as e:
        pass
    logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                collected_triples.append(parsed_triple)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()

        relation_description = description[index_of_colon + 1 :].strip()

        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict


def is_model_openai(model_name):
    return "gpt" in model_name


def generate_completion_transformers(
    input: list,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_token=256,
    answer_prepend="",
):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    chat_template_options = {"add_generation_prompt": True, "tokenize": False}
    if getattr(getattr(model, "config", None), "model_type", None) == "qwen3":
        chat_template_options["enable_thinking"] = False
    messages = tokenizer.apply_chat_template(input, **chat_template_options) + answer_prepend

    model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    generation = model.generate(**model_inputs, generation_config=generation_config)
    sequences = generation["sequences"]
    generated_ids = sequences[:, model_inputs["input_ids"].shape[1] :]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"[DEBUG] Standardize the extracted triples-: '{generated_texts}'")
    logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")

    # 如果是验证选项的场景，处理生成的文本
    if answer_prepend == "Answer: ":
        extracted_letter = extract_option_letter(generated_texts)
        if extracted_letter:
            generated_texts = extracted_letter
            # print(f"[DEBUG] 提取到的答案: '{generated_texts}'")
        else:
            print(f"[WARNING] 无法从文本中提取到有效的选项字母")

    # logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
    return generated_texts

def extract_option_letter(text: str) -> str:
    """
    从生成的文本中提取选项字母，使用多重匹配策略
    """
    # print(f"[DEBUG] 尝试从文本中提取选项字母: '{text}'")
    
    # 定义所有可能的匹配模式
    patterns = [
        # 1. 单个字母
        (r'^([A-Z])$', "单个字母"),
        # 2. "选项X"模式
        (r'选项\s*([A-Z])', "'选项X'模式"),
        # 3. "X选项"模式
        (r'([A-Z])\s*选项', "'X选项'模式"),
        # 4. "选择X"模式
        (r'选择\s*([A-Z])', "'选择X'模式"),
        # 5. Answer/答案模式
        (r'[Aa]nswer\s*[:：]\s*([A-Z])', "'Answer: X'模式"),
        (r'答案\s*[:：]\s*([A-Z])', "'答案: X'模式"),
        # 6. 开头字母
        (r'^([A-Z])[.,。，\s]', "开头字母"),
        # 7. 独立字母
        (r'[^A-Z]([A-Z])[^A-Z]', "独立字母"),
        # 8. "更合适"相关模式
        (r'([A-Z])\s*更合适', "'X更合适'模式"),
        # 9. 其他常见模式
        (r'选择选项\s*([A-Z])', "'选择选项X'模式"),
        (r'应选\s*([A-Z])', "'应选X'模式"),
    ]
    
    # 首先将文本转换为大写以统一处理
    text = text.upper()
    
    # 依次尝试所有模式
    for pattern, desc in patterns:
        match = re.search(pattern, text)
        if match:
            result = match.group(1)
            # print(f"[DEBUG] 使用{desc}匹配成功: '{result}'")
            return result
            
    # 如果上述模式都未匹配，尝试提取任何A-F范围内的字母
    letters = [c for c in text if c.isalpha() and 'A' <= c <= 'F']
    if letters:
        # print(f"[DEBUG] 提取到A-F范围内的第一个字母: '{letters[0]}'")
        return letters[0]
    
    # 最后尝试提取任何字母
    letters = [c for c in text if c.isalpha()]
    if letters:
        # print(f"[DEBUG] 提取到的第一个字母: '{letters[0]}'")
        return letters[0]
    
    print(f"[DEBUG] 无法从验证模型答案中提取到任何选项字母")
    return None




def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512):
    openai.api_key = os.environ["OPENAI_KEY"]
    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history
    while response is None:
        try:
            response = openai.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            time.sleep(5)
    logging.debug(f"Model: {model}\nPrompt:\n {messages}\n Result: {response.choices[0].message.content}")
    return response.choices[0].message.content
