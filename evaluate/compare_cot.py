"""Compare saved no-CoT and CoT results on the same extracted triples."""

import argparse
import json
from pathlib import Path

from evaluate.benchmark_metrics import ENTITY_ALIASES, evaluate


def normalized_triple(triple):
    if triple is None:
        return None
    return (ENTITY_ALIASES.get(triple[0], triple[0]), triple[1],
            ENTITY_ALIASES.get(triple[2], triple[2]))


def compare(baseline_path, cot_path, gold_path):
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    cot = json.loads(cot_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))["cases"]
    if len(baseline) != len(cot) or len(cot) != len(gold):
        raise ValueError("Case counts differ")
    changes = []
    decisions = []
    for case_index, (old_row, new_row, reference) in enumerate(zip(baseline, cot, gold)):
        for key in ("input_text", "oie", "schema_definition"):
            if old_row[key] != new_row[key]:
                raise ValueError(f"Case {case_index + 1} has different {key}; SC is not isolated")
        old_triples = old_row["schema_canonicalizaiton"]
        new_triples = new_row["schema_canonicalizaiton"]
        if len(old_triples) != len(new_triples):
            raise ValueError(f"Case {case_index + 1} has different triple counts")
        gold_triples = {normalized_triple(triple) for triple in reference["canonical"]}
        traces = new_row.get("canonicalization_reasoning", [])
        for trace in traces:
            extracted = trace["query_triplet"]
            triple_index = next(
                index for index, triple in enumerate(old_row["oie"]) if triple == extracted
            )
            decisions.append({
                "case": case_index + 1,
                "triple_index": triple_index,
                "source_relation": extracted[1],
                "baseline_relation": (old_triples[triple_index] or [None, None])[1],
                "cot_relation": (new_triples[triple_index] or [None, None])[1],
                "expected_relation": next((
                    triple[1] for triple in reference["canonical"]
                    if normalized_triple(triple)[0] == normalized_triple(extracted)[0]
                    and normalized_triple(triple)[2] == normalized_triple(extracted)[2]
                ), None),
                "selected_option": trace.get("selected_option"),
                "selected_candidate_rank": (
                    trace["candidate_relations"].index(trace["selected_relation"]) + 1
                    if trace.get("selected_relation") in trace["candidate_relations"] else None
                ),
                "family_override": trace.get("family_override"),
            })
        for triple_index, (old, new) in enumerate(zip(old_triples, new_triples)):
            if old == new:
                continue
            extracted = old_row["oie"][triple_index]
            trace = next(
                (item for item in traces if item["query_triplet"] == extracted), None
            )
            expected = [
                triple for triple in reference["canonical"]
                if normalized_triple(triple)[0] == normalized_triple(extracted)[0]
                and normalized_triple(triple)[2] == normalized_triple(extracted)[2]
            ]
            changes.append({
                "case": case_index + 1,
                "triple_index": triple_index,
                "extracted": extracted,
                "baseline": old,
                "cot": new,
                "expected": expected,
                "baseline_correct_with_entity_alias": normalized_triple(old) in gold_triples,
                "cot_correct_with_entity_alias": normalized_triple(new) in gold_triples,
                "cot_selected_option": trace.get("selected_option") if trace else None,
                "cot_family_override": trace.get("family_override") if trace else None,
                "cot_raw_output": trace.get("raw_output") if trace else None,
            })
    baseline_metrics = evaluate(baseline_path, gold_path)
    cot_metrics = evaluate(cot_path, gold_path)
    metric_keys = (
        "canonical_triple_exact", "canonical_triple_process_suffix_normalized",
        "canonical_relation_exact",
    )
    return {
        "cases": len(cot),
        "same_extraction_and_definitions": True,
        "baseline": {key: baseline_metrics[key] for key in metric_keys},
        "cot": {key: cot_metrics[key] for key in metric_keys},
        "cot_decisions": decisions,
        "changed_triples": changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--cot", required=True, type=Path)
    parser.add_argument("--gold", type=Path, default=Path(__file__).parent / "references" / "example_gold.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.cot, args.gold)
    report["baseline_result"] = str(args.baseline)
    report["cot_result"] = str(args.cot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline_f1": report["baseline"]["canonical_triple_process_suffix_normalized"]["f1"],
        "cot_f1": report["cot"]["canonical_triple_process_suffix_normalized"]["f1"],
        "changed_triples": len(report["changed_triples"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
