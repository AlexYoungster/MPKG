"""Operational quality checks for EDC output; no gold labels are assumed."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def read_schema(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or any(len(row) != 2 or not row[0].strip() or not row[1].strip() for row in rows):
        raise ValueError(f"Schema must have two nonempty CSV fields per row: {path}")
    labels = [row[0].strip() for row in rows]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate relation labels in schema: {path}")
    return set(labels)


def valid_triple(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(part, str) and part.strip() for part in value)
    )


def source_contains(text, entity):
    """Literal evidence heuristic; paraphrases and implicit subjects need review."""
    compact = lambda value: re.sub(r"[^\w]+", "", value.casefold())
    return bool(compact(entity)) and compact(entity) in compact(text)


def inspect_record(index, text, stage, schema, error=None):
    flags = []
    if not text.strip():
        flags.append("blank_input")
    if error:
        flags.append("pipeline_error")
    oie = stage.get("oie", []) if isinstance(stage, dict) else []
    canonical = stage.get("schema_canonicalizaiton", []) if isinstance(stage, dict) else []
    definitions = stage.get("schema_definition", {}) if isinstance(stage, dict) else {}
    if not isinstance(oie, list):
        oie, flags = [], flags + ["invalid_oie_container"]
    if not isinstance(canonical, list):
        canonical, flags = [], flags + ["invalid_canonical_container"]
    if not isinstance(definitions, dict):
        definitions, flags = {}, flags + ["invalid_definition_container"]
    if text.strip() and not oie and not error:
        flags.append("no_extracted_triples")
    if len(oie) != len(canonical) and not error:
        flags.append("triple_count_mismatch")

    triples = []
    seen = set()
    for position in range(max(len(oie), len(canonical))):
        raw = oie[position] if position < len(oie) else None
        final = canonical[position] if position < len(canonical) else None
        if not valid_triple(raw):
            flags.append("invalid_extracted_triple")
        elif raw[1] not in definitions:
            flags.append("missing_relation_definition")
        if final is None:
            if position < len(oie):
                flags.append("canonicalization_abstained")
            continue
        if not valid_triple(final):
            flags.append("invalid_canonical_triple")
            continue
        subject, relation, object_ = (part.strip() for part in final)
        if relation not in schema:
            flags.append("relation_outside_schema")
        if valid_triple(raw) and (subject != raw[0].strip() or object_ != raw[2].strip()):
            flags.append("entity_changed_during_canonicalization")
        triple_key = (subject, relation, object_)
        if triple_key in seen:
            flags.append("duplicate_canonical_triple")
        seen.add(triple_key)
        triples.append({
            "source_index": index,
            "source_line": index + 1,
            "triple_index": position,
            "subject": subject,
            "relation": relation,
            "object": object_,
            "relation_in_schema": relation in schema,
            "subject_verbatim": source_contains(text, subject),
            "object_verbatim": source_contains(text, object_),
        })

    if any(not item["subject_verbatim"] or not item["object_verbatim"] for item in triples):
        flags.append("entity_not_verbatim_in_source_review")
    blocking = [flag for flag in set(flags) if flag != "entity_not_verbatim_in_source_review"]
    record = {
        "index": index,
        "source_line": index + 1,
        "input_text": text,
        "status": "error" if error else "checked",
        "error": error,
        "oie_count": len(oie),
        "canonical_count": len(triples),
        "abstained_count": sum(item is None for item in canonical),
        "structural_and_schema_compliant": not blocking,
        "flags": sorted(set(flags)),
    }
    return record, triples


def summarize(records, triples, schema_size):
    flags = Counter(flag for record in records for flag in record["flags"])
    relations = Counter(item["relation"] for item in triples)
    nonblank = sum("blank_input" not in record["flags"] for record in records)
    checked = sum(record["status"] == "checked" and "blank_input" not in record["flags"] for record in records)
    compliant = sum(record["structural_and_schema_compliant"] and "blank_input" not in record["flags"] for record in records)
    extracted_count = sum(record["oie_count"] for record in records)
    return {
        "total_lines": len(records),
        "nonblank_lines": nonblank,
        "blank_lines": len(records) - nonblank,
        "checked_lines": checked,
        "pending_lines": sum(record["status"] == "pending" for record in records),
        "pipeline_error_lines": sum(record["status"] == "error" for record in records),
        "lines_with_extracted_triples": sum(record["oie_count"] > 0 for record in records),
        "lines_with_canonical_triples": sum(record["canonical_count"] > 0 for record in records),
        "structural_and_schema_compliant_lines": compliant,
        "processing_completion_rate": checked / nonblank if nonblank else 0.0,
        "structural_and_schema_compliance_rate": compliant / nonblank if nonblank else 0.0,
        "extracted_triples": extracted_count,
        "canonical_triples": len(triples),
        "canonicalization_retention_rate": len(triples) / extracted_count if extracted_count else 0.0,
        "canonicalization_abstentions": sum(record["abstained_count"] for record in records),
        "schema_relation_membership_rate": (
            sum(item["relation_in_schema"] for item in triples) / len(triples) if triples else 0.0
        ),
        "literal_entity_pair_rate": (
            sum(item["subject_verbatim"] and item["object_verbatim"] for item in triples) / len(triples)
            if triples else 0.0
        ),
        "schema_relation_count": schema_size,
        "relations_used": len(relations),
        "flag_counts": dict(sorted(flags.items())),
        "relation_counts": dict(sorted(relations.items())),
        "note": "Compliance checks only structure, schema membership and stage consistency. Literal source matching is a review signal. Semantic precision/recall requires labeled gold triples.",
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_reports(output_dir, inputs, stages, errors, schema):
    records, triples = [], []
    for index, text in enumerate(inputs):
        record, row_triples = inspect_record(index, text, stages.get(index), schema, errors.get(index))
        if index not in stages and index not in errors and text.strip():
            record["status"] = "pending"
            record["flags"] = ["pending"]
            record["structural_and_schema_compliant"] = False
        records.append(record)
        triples.extend(row_triples)
    summary = summarize(records, triples, len(schema))
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "records.jsonl", records)
    write_jsonl(output_dir / "triples.jsonl", triples)
    write_jsonl(output_dir / "review.jsonl", [record for record in records if record["flags"]])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path, help="Existing result_at_each_stage.json")
    parser.add_argument("--schema", required=True, type=Path, help="Two-column relation schema CSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    stages_list = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(stages_list, list):
        parser.error("Result must be a JSON array")
    inputs = [stage["input_text"].rstrip("\r\n") for stage in stages_list]
    stages = {index: stage for index, stage in enumerate(stages_list)}
    summary = write_reports(args.output_dir, inputs, stages, {}, read_schema(args.schema))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
