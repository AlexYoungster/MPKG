"""Extract all lines of a dataset with resumable EDC batches and quality reports."""

import argparse
import contextlib
import hashlib
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

from evaluate.compliance_report import read_schema, write_json, write_reports


ROOT = Path(__file__).resolve().parent


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "datasets" / "TestProcess.txt")
    parser.add_argument("--schema", type=Path, default=ROOT / "schemas" / "process_relations_en.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-records", type=int, help="Process the first N source lines for a trial run")
    parser.add_argument("--resume", action="store_true", help="Continue an existing matching output directory")
    parser.add_argument("--offline", action="store_true", help="Use models already present in the local Hugging Face cache")
    parser.add_argument("--sd-mode", choices=("model", "schema"), default="model",
                        help="Use model-derived abstract definitions or target-schema definitions with generic fallback")
    parser.add_argument("--oie-llm", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--sd-llm", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--sc-llm", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--embedder", default="intfloat/multilingual-e5-small")
    parser.add_argument("--oie-prompt", type=Path, default=ROOT / "prompt_templates" / "oie_template_en.txt")
    parser.add_argument("--oie-examples", type=Path, default=ROOT / "few_shot_examples" / "TestProcess" / "oie_few_shot_examples.txt")
    parser.add_argument("--sd-prompt", type=Path, default=ROOT / "prompt_templates" / "sd_template.txt")
    parser.add_argument("--sd-examples", type=Path, default=ROOT / "few_shot_examples" / "example" / "sd_few_shot_examples.txt")
    parser.add_argument("--sc-prompt", type=Path, default=ROOT / "prompt_templates" / "sc_template.txt")
    return parser


def edc_settings(args):
    return {
        "oie_llm": args.oie_llm,
        "oie_prompt_template_file_path": str(args.oie_prompt.resolve()),
        "oie_few_shot_example_file_path": str(args.oie_examples.resolve()),
        "sd_llm": args.sd_llm,
        "sd_prompt_template_file_path": str(args.sd_prompt.resolve()),
        "sd_few_shot_example_file_path": str(args.sd_examples.resolve()),
        "sc_llm": args.sc_llm,
        "sc_embedder": args.embedder,
        "sc_prompt_template_file_path": str(args.sc_prompt.resolve()),
        "sr_adapter_path": None,
        "sr_embedder": args.embedder,
        "oie_refine_prompt_template_file_path": str(ROOT / "prompt_templates" / "oie_r_template.txt"),
        "oie_refine_few_shot_example_file_path": str(ROOT / "few_shot_examples" / "example" / "oie_few_shot_refine_examples.txt"),
        "ee_llm": args.oie_llm,
        "ee_prompt_template_file_path": str(ROOT / "prompt_templates" / "ee_template.txt"),
        "ee_few_shot_example_file_path": str(ROOT / "few_shot_examples" / "example" / "ee_few_shot_examples.txt"),
        "em_prompt_template_file_path": str(ROOT / "prompt_templates" / "em_template.txt"),
        "target_schema_path": str(args.schema.resolve()),
        "refinement_iterations": 0,
        "enrich_schema": False,
        "loglevel": logging.WARNING,
    }


def manifest_data(args, all_lines):
    files = [args.input, args.schema, args.oie_prompt, args.oie_examples,
             args.sd_prompt, args.sd_examples, args.sc_prompt,
             ROOT / "batch_extract.py", ROOT / "evaluate" / "compliance_report.py",
             ROOT / "edc" / "edc_framework.py", ROOT / "edc" / "extract.py",
             ROOT / "edc" / "schema_definition.py", ROOT / "edc" / "schema_canonicalization.py"]
    return {
        "source": str(args.input.resolve()),
        "source_lines": len(all_lines),
        "selected_lines": min(len(all_lines), args.max_records) if args.max_records else len(all_lines),
        "files_sha256": {str(path.resolve()): file_digest(path) for path in files},
        "models": {
            "oie": args.oie_llm, "sd": args.sd_llm,
            "sc": args.sc_llm, "embedder": args.embedder, "sd_mode": args.sd_mode,
        },
        "chunk_size": args.chunk_size,
    }


def load_completed(chunks_dir, positions, inputs):
    first, last = positions[0] + 1, positions[-1] + 1
    for attempt_dir in sorted(chunks_dir.glob(f"{first:06d}_{last:06d}_attempt*"), reverse=True):
        result_path = attempt_dir / "iter0" / "result_at_each_stage.json"
        if not result_path.is_file():
            continue
        try:
            stages = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(stages, list) and len(stages) == len(positions)
                and all(isinstance(stage, dict) and
                        stage.get("input_text", "").rstrip("\r\n") == inputs[index]
                        for index, stage in zip(positions, stages))):
            return dict(zip(positions, stages))
    return None


def next_attempt_dir(chunks_dir, positions):
    first, last = positions[0] + 1, positions[-1] + 1
    prefix = f"{first:06d}_{last:06d}_attempt"
    numbers = [int(path.name[len(prefix):]) for path in chunks_dir.glob(prefix + "*")
               if path.name[len(prefix):].isdigit()]
    return chunks_dir / f"{prefix}{max(numbers, default=0) + 1:02d}"


def main(argv=None, edc_factory=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.chunk_size < 1 or (args.max_records is not None and args.max_records < 1):
        parser.error("--chunk-size and --max-records must be positive")
    for path in (args.input, args.schema, args.oie_prompt, args.oie_examples,
                 args.sd_prompt, args.sd_examples, args.sc_prompt):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    schema = read_schema(args.schema)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    all_lines = args.input.read_text(encoding="utf-8-sig").splitlines()
    inputs = all_lines[:args.max_records] if args.max_records else all_lines
    output_dir = args.output_dir or ROOT / "output" / f"batch_{args.input.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = manifest_data(args, all_lines)
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"No manifest to resume: {manifest_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            parser.error("Source, schema, prompts, models, limit, or chunk size changed since this run")
    else:
        if output_dir.exists():
            parser.error(f"Output directory already exists: {output_dir}; use --resume")
        output_dir.mkdir(parents=True)
        write_json(manifest_path, manifest)

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    stages, errors = {}, {}
    valid_positions = [index for index, text in enumerate(inputs) if text.strip()]
    edc = None

    def process(positions):
        nonlocal edc
        completed = load_completed(chunks_dir, positions, inputs)
        if completed is not None:
            stages.update(completed)
            return
        if len(positions) > 1:
            first, last = positions[0] + 1, positions[-1] + 1
            if list(chunks_dir.glob(f"{first:06d}_{last:06d}_attempt*")):
                middle = len(positions) // 2
                process(positions[:middle])
                process(positions[middle:])
                return
        attempt_dir = next_attempt_dir(chunks_dir, positions)
        if edc is None:
            if edc_factory is None:
                from edc.edc_framework import EDC
                if args.sd_mode == "schema":
                    from edc.schema_definition import SchemaDefiner

                    class SchemaFirstEDC(EDC):
                        def schema_definition(self, input_text_list, oie_triplets_list, free_model=False):
                            definitions = []
                            for triples in oie_triplets_list:
                                grouped = {}
                                for triple in triples:
                                    if len(triple) == 3:
                                        grouped.setdefault(triple[1], []).append(triple)
                                definitions.append({
                                    relation: self.schema.get(relation)
                                    or SchemaDefiner._generic_instance_definition(relation, examples)
                                    for relation, examples in grouped.items()
                                })
                            return definitions

                    edc = SchemaFirstEDC(**edc_settings(args))
                else:
                    edc = EDC(**edc_settings(args))
            else:
                edc = edc_factory(**edc_settings(args))
        try:
            with (output_dir / "batch.log").open("a", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    edc.extract_kg([inputs[index] for index in positions], str(attempt_dir))
            completed = load_completed(chunks_dir, positions, inputs)
            if completed is None:
                raise ValueError("EDC did not write a complete matching result file")
            stages.update(completed)
        except Exception as exc:
            write_json(attempt_dir.with_suffix(".error.json"), {
                "source_lines": [index + 1 for index in positions],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            if len(positions) == 1:
                errors[positions[0]] = f"{type(exc).__name__}: {exc}"
            else:
                middle = len(positions) // 2
                process(positions[:middle])
                process(positions[middle:])

    write_reports(output_dir, inputs, stages, errors, schema)
    for start in range(0, len(valid_positions), args.chunk_size):
        positions = valid_positions[start:start + args.chunk_size]
        process(positions)
        summary = write_reports(output_dir, inputs, stages, errors, schema)
        print(f"Completed {summary['checked_lines']}/{summary['nonblank_lines']} lines; "
              f"compliant {summary['structural_and_schema_compliant_lines']}; "
              f"errors {summary['pipeline_error_lines']}", flush=True)
    return write_reports(output_dir, inputs, stages, errors, schema)


if __name__ == "__main__":
    main()
