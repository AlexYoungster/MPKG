from argparse import ArgumentParser
from edc.edc_framework import EDC
import os
import logging
import datetime
import copy
import json
from pathlib import Path


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def replay_schema_canonicalization(edc, source_path, output_dir):
    """Run only SC on saved OIE and SD results for a controlled comparison."""
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    if not isinstance(source, list) or not all(
        isinstance(row, dict) and "input_text" in row
        and isinstance(row.get("oie"), list)
        and isinstance(row.get("schema_definition"), dict)
        for row in source
    ):
        raise ValueError("Replay input must be result_at_each_stage.json")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    inputs = [row["input_text"] for row in source]
    extracted = [row["oie"] for row in source]
    definitions = [row["schema_definition"] for row in source]
    canonical, candidates = edc.schema_canonicalization(inputs, extracted, definitions)
    rows = copy.deepcopy(source)
    for index, row in enumerate(rows):
        row["schema_canonicalizaiton"] = canonical[index]
        row["canonicalization_candidates"] = str(candidates[index])
        if edc.sc_cot:
            row["canonicalization_reasoning"] = edc.last_cot_trace_by_entry[index]
        else:
            row.pop("canonicalization_reasoning", None)
    iteration_dir = destination / "iter0"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "result_at_each_stage.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (iteration_dir / "canon_kg.txt").write_text(
        "\n".join(str([triple for triple in text_triples if triple is not None])
                  for text_triples in canonical), encoding="utf-8"
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    DEFAULT_LLM = "Qwen/Qwen3-1.7B"
    DEFAULT_EMBEDDER = "intfloat/multilingual-e5-small"
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # OIE module setting
    parser.add_argument(
        "--oie_llm", default=DEFAULT_LLM, help="LLM used for open information extraction."
    )
    parser.add_argument(
        "--oie_prompt_template_file_path",
        default="./prompt_templates/oie_template.txt",
        help="Promp template used for open information extraction.",
    )
    parser.add_argument(
        "--oie_few_shot_example_file_path",
        default="./few_shot_examples/example/oie_few_shot_examples.txt",
        help="Few shot examples used for open information extraction.",
    )

    # Schema Definition setting
    parser.add_argument(
        "--sd_llm", default=DEFAULT_LLM, help="LLM used for schema definition."
    )
    parser.add_argument(
        "--sd_prompt_template_file_path",
        default="./prompt_templates/sd_template.txt",
        help="Prompt template used for schema definition.",
    )
    parser.add_argument(
        "--sd_few_shot_example_file_path",
        default="./few_shot_examples/example/sd_few_shot_examples.txt",
        help="Few shot examples used for schema definition.",
    )

    # Schema Canonicalization setting
    # CoT mode is opt-in so existing runs keep the short-answer verifier.
    parser.add_argument(
        "--sc_llm",
        default=DEFAULT_LLM,
        help="LLM used for schema canonicaliztion verification.",
    )
    parser.add_argument(
        "--sc_embedder", default=DEFAULT_EMBEDDER, help="Embedder used for schema canonicalization. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--sc_prompt_template_file_path",
        default="./prompt_templates/sc_template.txt",
        help="Prompt template used for schema canonicalization verification.",
    )
    parser.add_argument("--sc_cot", action="store_true", help="Use CoT verification for schema canonicalization.")
    parser.add_argument("--sc_cot_max_tokens", type=int, default=256,
                        help="Maximum answer tokens for the CoT verifier.")
    parser.add_argument("--sc_replay_result_path", default=None,
                        help="Replay SC from a saved result_at_each_stage.json without repeating OIE and SD.")

    # Refinement setting
    parser.add_argument("--sr_adapter_path", default=None, help="Path to adapter of schema retriever.")
    parser.add_argument(
        "--sr_embedder", default=DEFAULT_EMBEDDER, help="Embedding model used for schema retriever. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--oie_refine_prompt_template_file_path",
        default="./prompt_templates/oie_r_template.txt",
        help="Prompt template used for refined open information extraction.",
    )
    parser.add_argument(
        "--oie_refine_few_shot_example_file_path",
        default="./few_shot_examples/example/oie_few_shot_refine_examples.txt",
        help="Few shot examples used for refined open information extraction.",
    )
    parser.add_argument(
        "--ee_llm", default=DEFAULT_LLM, help="LLM used for entity extraction."
    )
    parser.add_argument(
        "--ee_prompt_template_file_path",
        default="./prompt_templates/ee_template.txt",
        help="Prompt templated used for entity extraction.",
    )
    parser.add_argument(
        "--ee_few_shot_example_file_path",
        default="./few_shot_examples/example/ee_few_shot_examples.txt",
        help="Few shot examples used for entity extraction.",
    )
    parser.add_argument(
        "--em_prompt_template_file_path",
        default="./prompt_templates/em_template.txt",
        help="Prompt template used for entity merging.",
    )

    # Input setting
    parser.add_argument(
        "--input_text_file_path",
        default="./datasets/example.txt",
        help="File containing input texts to extract KG from, each line contains one piece of text.",
    )
    parser.add_argument(
        "--target_schema_path",
        default="./schemas/example_schema.csv",
        help="File containing the target schema to align to.",
    )
    parser.add_argument("--refinement_iterations", default=0, type=int, help="Number of iteration to run.")
    parser.add_argument(
        "--enrich_schema",
        action="store_true",
        help="Whether un-canonicalizable relations should be added to the schema.",
    )
    # Output setting
    parser.add_argument("--output_dir", default=f"./output/tmp_{current_time}", help="Directory to output to.")
    parser.add_argument("--logging_verbose", action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument("--logging_debug", action="store_const", dest="loglevel", const=logging.DEBUG)

    args = parser.parse_args()
    args = vars(args)
    template_name = Path(args["sc_prompt_template_file_path"]).name
    if template_name == "sc_template_cot.txt":
        args["sc_cot"] = True
    elif args["sc_cot"] and template_name == "sc_template.txt":
        args["sc_prompt_template_file_path"] = "./prompt_templates/sc_template_cot.txt"
    edc = EDC(**args)
    

    if args["sc_replay_result_path"]:
        replay_schema_canonicalization(edc, args["sc_replay_result_path"], args["output_dir"])
    else:
        input_text_list = open(args["input_text_file_path"], "r").readlines()
        output_kg = edc.extract_kg(
            input_text_list,
            args["output_dir"],
            refinement_iterations=args["refinement_iterations"],
        )
