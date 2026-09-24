"""Exact, reproducible metrics for the checked-in example benchmark."""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

RELATION_ALIASES = {"制造材料": "材料"}
ENTITY_ALIASES = {"内圆磨": "内圆磨加工"}


def _triples(values):
    return [tuple(value) for value in values if value is not None and len(value) == 3]


def _score(gold, predicted):
    gold_counts = Counter(gold)
    predicted_counts = Counter(predicted)
    true_positive = sum((gold_counts & predicted_counts).values())
    precision = true_positive / sum(predicted_counts.values()) if predicted_counts else 0.0
    recall = true_positive / sum(gold_counts.values()) if gold_counts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "correct": true_positive,
        "predicted": sum(predicted_counts.values()),
        "gold": sum(gold_counts.values()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _load_candidates(raw):
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return raw if isinstance(raw, list) else []


def _normalized_entity(value):
    """Normalize the documented inner-grinding name variant in this benchmark."""
    return ENTITY_ALIASES.get(value, value)


def _normalized_triples(values):
    return [
        (_normalized_entity(s), RELATION_ALIASES.get(r, r), _normalized_entity(o))
        for s, r, o in _triples(values)
    ]


def evaluate(result_path: Path, gold_path: Path):
    results = json.loads(result_path.read_text(encoding="utf-8"))
    reference = json.loads(gold_path.read_text(encoding="utf-8"))
    cases = reference["cases"]
    if len(results) != len(cases):
        raise ValueError(f"case count differs: result={len(results)}, gold={len(cases)}")

    gold_extraction = []
    predicted_extraction = []
    gold_canonical = []
    predicted_canonical = []
    gold_entity_pairs = []
    predicted_entity_pairs = []
    gold_entity_pairs_normalized = []
    predicted_entity_pairs_normalized = []
    gold_relations = []
    predicted_relations = []
    candidate_hits = {5: 0, 8: 0}
    candidate_total = 0

    for result, case in zip(results, cases):
        expected_open = _triples(case["extraction"])
        expected_canonical = _triples(case["canonical"])
        raw_oie = _triples(result.get("oie", []))
        raw_canonical = _triples(result.get("schema_canonicalizaiton", []))
        gold_extraction.extend(expected_open)
        predicted_extraction.extend(raw_oie)
        gold_canonical.extend(expected_canonical)
        predicted_canonical.extend(raw_canonical)
        gold_entity_pairs.extend((s, o) for s, _, o in expected_open)
        predicted_entity_pairs.extend((s, o) for s, _, o in raw_oie)
        gold_entity_pairs_normalized.extend(
            (_normalized_entity(s), _normalized_entity(o)) for s, _, o in expected_open
        )
        predicted_entity_pairs_normalized.extend(
            (_normalized_entity(s), _normalized_entity(o)) for s, _, o in raw_oie
        )
        gold_relations.extend(relation for _, relation, _ in expected_open)
        predicted_relations.extend(relation for _, relation, _ in raw_oie)

        per_triple_candidates = _load_candidates(result.get("canonicalization_candidates", []))
        for expected_source in expected_open:
            target = next(
                (
                    triple
                    for triple in expected_canonical
                    if _normalized_entity(triple[0]) == _normalized_entity(expected_source[0])
                    and _normalized_entity(triple[2]) == _normalized_entity(expected_source[2])
                ),
                None,
            )
            if target is None or expected_source[1] == target[1]:
                continue
            candidate_total += 1
            matching_index = next(
                (
                    index
                    for index, source in enumerate(raw_oie)
                    if _normalized_entity(source[0]) == _normalized_entity(expected_source[0])
                    and _normalized_entity(source[2]) == _normalized_entity(expected_source[2])
                    and RELATION_ALIASES.get(source[1], source[1])
                    == RELATION_ALIASES.get(expected_source[1], expected_source[1])
                ),
                None,
            )
            candidate_dict = (
                per_triple_candidates[matching_index]
                if matching_index is not None and matching_index < len(per_triple_candidates)
                else {}
            )
            ordered_candidates = list(candidate_dict) if isinstance(candidate_dict, dict) else []
            for cutoff in candidate_hits:
                if target[1] in ordered_candidates[:cutoff]:
                    candidate_hits[cutoff] += 1

    entity_score = _score(gold_entity_pairs, predicted_entity_pairs)
    normalized_entity_score = _score(
        gold_entity_pairs_normalized, predicted_entity_pairs_normalized
    )
    relation_score = _score(gold_relations, predicted_relations)
    extraction_score = _score(gold_extraction, predicted_extraction)
    extraction_normalized_score = _score(
        _normalized_triples(gold_extraction), _normalized_triples(predicted_extraction)
    )
    canonical_score = _score(gold_canonical, predicted_canonical)
    canonical_normalized_score = _score(
        _normalized_triples(gold_canonical), _normalized_triples(predicted_canonical)
    )
    candidate_recall = {
        f"recall_at_{cutoff}": candidate_hits[cutoff] / candidate_total if candidate_total else 0.0
        for cutoff in candidate_hits
    }
    return {
        "cases": len(cases),
        "extraction_entity_pair_exact": entity_score,
        "extraction_entity_pair_process_suffix_normalized": normalized_entity_score,
        "extraction_relation_exact": relation_score,
        "extraction_triple_exact": extraction_score,
        "extraction_triple_entity_and_relation_alias_normalized": extraction_normalized_score,
        "canonical_triple_exact": canonical_score,
        "canonical_triple_process_suffix_normalized": canonical_normalized_score,
        "canonical_relation_exact": _score(
            [relation for _, relation, _ in gold_canonical],
            [relation for _, relation, _ in predicted_canonical],
        ),
        "candidate_recall": {"mapped_relations": candidate_total, **candidate_recall},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path, help="result_at_each_stage.json path")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(__file__).parent / "references" / "example_gold.json",
        help="gold JSON path",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args.result, args.gold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
