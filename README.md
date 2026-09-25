# MPKG: Manufacturing Process Knowledge Graph

An LLM-driven framework for automatic knowledge graph construction from manufacturing process texts. MPKG adapts the Extract-Define-Canonicalize (EDC) pipeline to the manufacturing domain, enabling structured knowledge extraction from unstructured machining process descriptions.

## Overview

MPKG extracts structured knowledge graphs from manufacturing process texts through a multi-stage pipeline:

1. **Open Information Extraction (OIE)** — Extracts raw relation triplets `[subject, relation, object]` from input text using LLM with few-shot prompting
2. **Schema Definition (SD)** — Generates natural language definitions for each extracted relation
3. **Schema Canonicalization (SC)** — Aligns extracted relations to a predefined target schema using embedding similarity retrieval + LLM verification
4. **Iterative Refinement** (optional) — Refines extraction results through entity extraction, entity merging, and schema-guided re-extraction

```
Input Text ──► OIE ──► SD ──► SC ──► Canonicalized KG
                 ▲                        │
                 └── Refinement (EE+EM) ──┘
```

## Project Structure

```
MPKG/
├── run.py                          # Main entry point
├── run.sh                          # Example run script
├── edc/                            # Core framework
│   ├── edc_framework.py            # EDC pipeline orchestrator
│   ├── extract.py                  # Open Information Extraction module
│   ├── schema_definition.py        # Schema Definition module
│   ├── schema_canonicalization.py  # Schema Canonicalization module
│   ├── schema_canonicalization_cot.py  # CoT-enhanced canonicalization
│   ├── entity_extraction.py        # Entity Extraction for refinement
│   ├── schema_retriever.py         # Schema Retriever for refinement
│   └── utils/
│       ├── llm_utils.py            # LLM inference & parsing utilities
│       └── e5_mistral_utils.py     # E5-Mistral embedding utilities
├── prompt_templates/               # Prompt templates for each stage
│   ├── oie_template.txt            # OIE prompt
│   ├── sd_template.txt             # Schema Definition prompt
│   ├── sc_template.txt             # Schema Canonicalization prompt
│   ├── sc_template_cot.txt         # CoT Schema Canonicalization prompt
│   ├── oie_r_template.txt          # Refined OIE prompt
│   ├── ee_template.txt             # Entity Extraction prompt
│   └── em_template.txt             # Entity Merging prompt
├── few_shot_examples/              # Few-shot examples for each dataset
│   ├── example/                    # Manufacturing domain examples
│   ├── webnlg/                     # WebNLG benchmark examples
│   ├── wiki-nre/                   # Wiki-NRE benchmark examples
│   └── rebel/                      # REBEL benchmark examples
├── schemas/                        # Target schema definitions
│   ├── example_schema.csv          # Manufacturing process schema (CSV)
│   ├── Mechanical_Schema_Ch.csv    # Full mechanical schema (Chinese)
│   ├── Mechanical_Schema_En.csv    # Full mechanical schema (English)
│   ├── OWL/                        # OWL ontology representations
│   └── code_style/                 # Code-style schema representations
├── datasets/                       # Input text datasets
│   ├── example.txt                 # Manufacturing process examples (Chinese)
│   └── TestProcess.txt             # Machining process texts (English)
├── evaluate/                       # Evaluation tools
│   └── evaluation_script.py        # Triplet evaluation script
├── output/                         # Pipeline output results
├── collect_schema_retrieval_data.py  # Schema retriever training data collection
├── schema_canonicalization.py      # Standalone canonicalization script
├── environment.yml                 # Conda environment specification
└── LICENSE                         # MIT License
```

## Installation

### Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA support (recommended; `run.sh` loads Qwen3-1.7B in 8-bit)
- Git Bash on Windows when running `run.sh` from Windows

### Setup

```bash
# Clone the repository
git clone https://github.com/NKU-IIPLab/MPKG.git
cd MPKG

# Option A: use a Python virtual environment
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Option B: use Conda
conda env create -f environment.yml
conda activate edc

# Run from Git Bash (the script also detects the local .venv)
bash run.sh
```

### Key Dependencies

- `transformers` — HuggingFace model loading and inference
- `sentence-transformers` — Embedding models for schema retrieval
- `openai` — OpenAI API support (optional)
- `torch` — PyTorch backend
- `numpy`, `pandas` — Data processing

## Usage

### Quick Start

```bash
# Run with default settings on the example dataset
bash run.sh
```

### Batch extraction and compliance report

`batch_extract.py` processes every nonblank line of `datasets/TestProcess.txt` (400 English texts by default). It uses a relation schema rather than the mechanical concept taxonomy. Each batch writes its stage result under `chunks/`; completed batches are reused on resume. A failed batch is split to isolate the failing source line.

From PowerShell in the project directory:

```powershell
.venv\Scripts\python.exe -X utf8 batch_extract.py --input datasets\TestProcess.txt --schema schemas\process_relations_en.csv --sd-mode schema --output-dir output\testprocess_full
```

If interrupted, rerun the same command with `--resume` appended. The script checks the source, schema, prompts, models, limit and batch size against `manifest.json` before resuming. Use `--chunk-size 10` to set the batch size (10 is the default), or `--max-records 1` for a model smoke test in a **separate** output directory.

The default `--sd-mode model` asks Qwen to generalize relation definitions from the extracted instances. The full-dataset command uses `--sd-mode schema`: it uses the target schema's general definitions for matching labels and abstracts unmatched labels from their extracted instances, avoiding the extra SD generation pass. Add `--offline` when both models are already cached locally and Hugging Face is unreachable. Use the same `--sd-mode` value when resuming.

For the three Chinese example texts, supply the corresponding schema and OIE prompts:

```powershell
.venv\Scripts\python.exe -X utf8 batch_extract.py --input datasets\example.txt --schema schemas\example_schema.csv --oie-prompt prompt_templates\oie_template.txt --oie-examples few_shot_examples\example\oie_few_shot_examples.txt --output-dir output\example_batch
```

Batch output files:

| File | Contents |
|------|----------|
| `manifest.json` | Source and configuration fingerprints for safe resume |
| `chunks/*/iter0/result_at_each_stage.json` | Raw extraction, definitions, candidates and canonicalization for each batch |
| `records.jsonl` | One entry per original source line, including line number, counts and quality flags |
| `triples.jsonl` | Canonical triples with source line and literal source-match indicators |
| `review.jsonl` | Rows needing review, including blank input, pipeline errors and abstentions |
| `summary.json` | Totals, rates, issue counts and relation frequencies |
| `batch.log` | Model and pipeline output |

The reported **structural and schema compliance rate** checks triple shape, nonempty fields, relation membership in the selected schema, consistency between extraction and canonicalization, and duplicate triples. Literal subject/object matching is a separate review signal because valid paraphrases and implicit subjects may not appear verbatim. These checks cannot establish semantic correctness or extraction recall. Use manually labeled reference triples and `evaluate/benchmark_metrics.py` when accuracy metrics are required.

To inspect an existing stage result without rerunning a model:

```powershell
.venv\Scripts\python.exe -X utf8 evaluate\compliance_report.py --result output\codex_validated_20260923\iter0\result_at_each_stage.json --schema schemas\example_schema.csv --output-dir output\example_compliance_report
```

### Custom Run

```bash
python run.py \
    --oie_llm <path-to-llm> \
    --sd_llm <path-to-llm> \
    --sc_llm <path-to-llm> \
    --sc_embedder <path-to-embedding-model> \
    --input_text_file_path ./datasets/example.txt \
    --target_schema_path ./schemas/example_schema.csv \
    --output_dir ./output/my_experiment \
    --logging_verbose
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--oie_llm` | LLM for Open Information Extraction | `Qwen/Qwen3-1.7B` |
| `--sd_llm` | LLM for Schema Definition | `Qwen/Qwen3-1.7B` |
| `--sc_llm` | LLM for Schema Canonicalization verification | `Qwen/Qwen3-1.7B` |
| `--sc_embedder` | Sentence Transformer for schema retrieval | `intfloat/multilingual-e5-small` |
| `--sc_cot` | Enable CoT for relation verification | `False` |
| `--sc_cot_max_tokens` | Maximum verifier output tokens in CoT mode | `256` |
| `--sc_replay_result_path` | Reuse saved OIE and SD results to evaluate SC alone | — |
| `--sr_embedder` | Embedding model for Schema Retriever (required when `--refinement_iterations > 0`) | — |
| `--ee_llm` | LLM for Entity Extraction (required when `--refinement_iterations > 0`) | — |
| `--input_text_file_path` | Input text file (one text per line) | `./datasets/example.txt` |
| `--target_schema_path` | Target schema CSV file (relation, definition) | `./schemas/example_schema.csv` |
| `--refinement_iterations` | Number of refinement iterations | `0` |
| `--enrich_schema` | Add un-canonicalizable relations to schema | `False` |
| `--output_dir` | Output directory | `./output/tmp_<timestamp>` |
| `--logging_verbose` | Enable INFO-level logging | — |
| `--logging_debug` | Enable DEBUG-level logging | — |
| `--sr_adapter_path` | Path to optional adapter for Schema Retriever | `None` |

> **Note:** Prompt templates and few-shot examples for each stage can be overridden via `--<stage>_prompt_template_file_path` and `--<stage>_few_shot_example_file_path` arguments (e.g. `--oie_prompt_template_file_path`, `--sd_few_shot_example_file_path`). Defaults point to files under `prompt_templates/` and `few_shot_examples/example/`.

`run.sh` uses the same defaults. Override `OIE_LLM`, `SD_LLM`, `SC_LLM`, `SC_EMBEDDER`, `DATASET`, or `OUTPUT_DIR` in the shell environment to customize a run. It defaults `HF_ENDPOINT` to `https://hf-mirror.com` for networks that cannot reach Hugging Face directly; set `HF_ENDPOINT=https://huggingface.co` to override it. The script allows longer Hugging Face metadata and download timeouts. Model files are cached under `.cache/models`; set `MPKG_MODEL_CACHE` to use another location.

### CoT-Enhanced Canonicalization

Use `--sc_cot` to select the CoT template automatically. Pointing `--sc_prompt_template_file_path` to `sc_template_cot.txt` also enables CoT for compatibility. This mode keeps the standard candidate retriever and relation type checks, allows a longer verifier answer, reads the explicit final option, and saves the verifier's text in `canonicalization_reasoning`:

```bash
python run.py \
    --sc_cot \
    --sc_prompt_template_file_path ./prompt_templates/sc_template_cot.txt \
    --output_dir ./output/example_cot
```

For a controlled comparison that holds OIE triples and SD definitions fixed, replay only SC from a previous run:

```bash
python run.py \
    --sc_cot \
    --sc_prompt_template_file_path ./prompt_templates/sc_template_cot.txt \
    --sc_replay_result_path ./output/codex_validated_20260923/iter0/result_at_each_stage.json \
    --output_dir ./output/example_cot_replay
python -m evaluate.compare_cot \
    --baseline ./output/codex_validated_20260923/iter0/result_at_each_stage.json \
    --cot ./output/example_cot_replay/iter0/result_at_each_stage.json \
    --output ./evaluate/reports/example_cot_comparison.json
```

### Supported Models

- **LLM**: Any HuggingFace causal LM (e.g., Qwen, Mistral-7B-Instruct) or OpenAI GPT models
- **Embedder**: Any Sentence Transformer model (e.g., BGE, E5-Mistral-7B-Instruct)

## Target Schema Format

The target schema is a CSV file where each row defines a relation. There is **no header row**; each line follows the format `relation_name,relation_definition`:

```csv
材料,主体零件由指定的材料构成-如"钛合金零件-材料-T-6A1-4V"
加工方法,主体零件通过指定的加工方法进行处理-如"精密轴套-加工方法-车削"
表面粗糙度要求,主体零件需要达到指定的表面质量要求-如"钛合金零件-表面粗糙度要求-Ra1.6"
```

The repository also provides schema representations in OWL ontology format and code-style format under `schemas/`.

## Output Format

For each iteration, the pipeline outputs:

- `result_at_each_stage.json` — Detailed results at each pipeline stage
- `canon_kg.txt` — Final canonicalized knowledge graph triplets

`result_at_each_stage.json` is a JSON array where each element corresponds to one input text:

```json
[
  {
    "index": 0,
    "input_text": "零件T-6A1-4V由钛合金组成...",
    "entity_hint": "",
    "relation_hint": "",
    "oie": [
      ["零件T-6A1-4V", "材料", "钛合金"],
      ["车削加工", "主轴转速", "300rpm"]
    ],
    "schema_definition": {
      "材料": "主体实体由客体实体构成或组成。",
      "主轴转速": "在进行主体实体的过程中，主轴的旋转速度为客体实体所规定的数值。"
    },
    "canonicalization_candidates": "[{'使用刀具': 0.74, '属于（工艺）': 0.65}, ...]",
    "schema_canonicalizaiton": [
      ["零件T-6A1-4V", "材料", "钛合金"],
      ["车削加工", "主轴转速", "300rpm"]
    ]
  }
]
```

`canon_kg.txt` contains the final triplets, one per line:

```
["零件T-6A1-4V", "材料", "钛合金"]
["车削加工", "使用刀具", "CBN刀具"]
["车削加工", "主轴转速", "300rpm"]
```

## Evaluation

```bash
python evaluate/evaluation_script.py \
    --edc_output /path/to/canon_kg.txt \
    --reference /path/to/reference.txt \
    --max_length_diff 5
```

The script reports **Precision**, **Recall**, and **F1** for triplet-level evaluation using [nervaluate](https://github.com/MantisAI/nervaluate). The evaluation script is adapted from the [WebNLG evaluation script](https://github.com/WebNLG/WebNLG-Text-to-triples).

For the checked-in three-text example benchmark, `evaluate/benchmark_metrics.py` reports exact extraction and canonical relation/triple scores, candidate Recall@5, and a separately labeled score normalized for the documented `内圆磨`/`内圆磨加工` entity variant. The manually reviewed references are in `evaluate/references/example_gold.json`; the baseline-to-improved run comparison is in `evaluate/reports/example_benchmark_comparison.json`:

```bash
python evaluate/benchmark_metrics.py --result ./output/<run-directory>/iter0/result_at_each_stage.json
```

## Knowledge Graph Q&A System Test
```shell
PS F:\MPKG> .venv\Scripts\python.exe -X utf8 qa_backend.py ask --db output\testprocess_graph_20260924.sqlite --question '面铣削先粗后精时，两道工序的切深、精加工进给和最终平面度分别是多少？' --offline
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.78s/it]
`generation_config` default values have been modified to match model-specific defaults: {'do_sample': True, 'temperature': 0.6, 'top_k': 20, 'top_p': 0.95, 'bos_token_id': 151643}. If this is not desired, please set these values explicitly.
{
  "question": "面铣削先粗后精时，两道工序的切深、精加工进给和最终平面度分别是多少？",
  "retrieval": {
    "mode": "translated_exact_entity",
    "anchor": "face milling",
    "ambiguous": true,
    "candidate_documents": 3,
    "translated_query": "Face milling with rough and finish machining, the depth of cut, finish feed, and final flatness are respectively 4.0 mm, 0.01 mm, and 0.05 mm."
  },
  "evidence": {
    "documents": [
      {
        "dataset_id": "source_005795735dd9f26011bb",
        "source_index": 0,
        "source_line": 1,
        "input_text": "The face milling operation was performed at a cutting speed of 180 m/min with a feed rate of 0.2 mm/rev and cutting depth of 2.5 mm using a carbide tool.",
        "status": "checked",
        "flags": [
          "canonicalization_abstained"
        ],
        "id": "D1"
      },
      {
        "dataset_id": "source_005795735dd9f26011bb",
        "source_index": 65,
        "source_line": 66,
        "input_text": "A precision face milling operation on aluminum housings employed a PCD cutter at 800 m/min, with high-pressure through-tool coolant at 80 bar and feed per tooth of 0.15 mm.",
        "status": "checked",
        "flags": [
          "canonicalization_abstained"
        ],
        "id": "D2"
      },
      {
        "dataset_id": "source_005795735dd9f26011bb",
        "source_index": 117,
        "source_line": 118,
        "input_text": "Face milling of aluminum plates specified high-speed machining parameters with 0.15 mm tooth feed, implementing climb cutting for all passes.",
        "status": "checked",
        "flags": [
          "canonicalization_abstained"
        ],
        "id": "D3"
      }
    ],
    "facts": [
      {
        "triple_index": 0,
        "subject": "face milling operation",
        "relation": "Operation",
        "object": "face milling",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E1",
        "document_id": "D1",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 1
      },
      {
        "triple_index": 1,
        "subject": "face milling",
        "relation": "Cutting Speed",
        "object": "180 m/min",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E2",
        "document_id": "D1",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 1
      },
      {
        "triple_index": 2,
        "subject": "face milling",
        "relation": "Feed Rate",
        "object": "0.2 mm/rev",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E3",
        "document_id": "D1",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 1
      },
      {
        "triple_index": 4,
        "subject": "face milling",
        "relation": "Tool",
        "object": "carbide tool",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E4",
        "document_id": "D1",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 1
      },
      {
        "triple_index": 0,
        "subject": "precision face milling operation",
        "relation": "Operation",
        "object": "face milling",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E5",
        "document_id": "D2",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 66
      },
      {
        "triple_index": 1,
        "subject": "face milling",
        "relation": "Tool",
        "object": "PCD cutter",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E6",
        "document_id": "D2",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 66
      },
      {
        "triple_index": 3,
        "subject": "face milling",
        "relation": "Coolant",
        "object": "high-pressure through-tool coolant",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E7",
        "document_id": "D2",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 66
      },
      {
        "triple_index": 4,
        "subject": "face milling",
        "relation": "Coolant Pressure",
        "object": "80 bar",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E8",
        "document_id": "D2",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 66
      },
      {
        "triple_index": 0,
        "subject": "face milling",
        "relation": "Tool",
        "object": "cutting",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E9",
        "document_id": "D3",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 118
      },
      {
        "triple_index": 1,
        "subject": "face milling",
        "relation": "Feed Rate",
        "object": "0.15 mm",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E10",
        "document_id": "D3",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 118
      }
    ]
  },
  "answer": "该名称对应多条加工记录，现有问题无法确定唯一工件及其参数。请补充工件全称、工序或原文行号。",
  "evidence_explanation": "候选出处：[D1]：“The face milling operation was performed at a cutting speed of 180 m/min with a feed rate of 0.2 mm/rev and cutting depth of 2.5 mm using a carbide tool.”、[D2]：“A precision face milling operation on aluminum housings employed a PCD cutter at 800 m/min, with high-pressure through-tool coolant at 80 bar and feed per tooth of 0.15 mm.”、[D3]：“Face milling of aluminum plates specified high-speed machining parameters with 0.15 mm tooth feed, implementing climb cutting for all passes.”"
}
PS F:\MPKG> .venv\Scripts\python.exe -X utf8 qa_backend.py ask --db output\testprocess_graph_20260924.sqlite --question '通孔钻削按什么顺序使用中心钻、先导钻和最终尺寸钻？' --offline
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.44s/it]
`generation_config` default values have been modified to match model-specific defaults: {'do_sample': True, 'temperature': 0.6, 'top_k': 20, 'top_p': 0.95, 'bos_token_id': 151643}. If this is not desired, please set these values explicitly.
{
  "question": "通孔钻削按什么顺序使用中心钻、先导钻和最终尺寸钻？",
  "retrieval": {
    "mode": "translated_exact_entity",
    "anchor": "final size drill",
    "ambiguous": false,
    "candidate_documents": 1,
    "translated_query": "Drilling holes in sequence using center drill, pilot drill, and final size drill."
  },
  "evidence": {
    "documents": [
      {
        "dataset_id": "source_005795735dd9f26011bb",
        "source_index": 73,
        "source_line": 74,
        "input_text": "A drilling operation for through holes required center drilling first, followed by pilot drilling at 0.3 times the final diameter, and then finishing with the final size drill.",
        "status": "checked",
        "flags": [
          "canonicalization_abstained",
          "entity_not_verbatim_in_source_review"
        ],
        "id": "D1"
      }
    ],
    "facts": [
      {
        "triple_index": 3,
        "subject": "drilling operation",
        "relation": "Tool",
        "object": "final size drill",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E1",
        "document_id": "D1",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 74
      },
      {
        "triple_index": 4,
        "subject": "drilling operation",
        "relation": "Tool",
        "object": "final diameter drill",
        "subject_verbatim": 1,
        "object_verbatim": 0,
        "type_warning": false,
        "id": "E2",
        "document_id": "D1",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 74
      },
      {
        "triple_index": 5,
        "subject": "drilling operation",
        "relation": "Tool",
        "object": "pilot drill",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E3",
        "document_id": "D1",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 74
      },
      {
        "triple_index": 6,
        "subject": "drilling operation",
        "relation": "Tool",
        "object": "center drill",
        "subject_verbatim": 1,
        "object_verbatim": 1,
        "type_warning": false,
        "id": "E4",
        "document_id": "D1",
        "dataset_id": "source_005795735dd9f26011bb",
        "source_line": 74
      }
    ]
  },
  "answer": "通孔钻削的顺序是先使用中心钻，接着使用先导钻，最后使用最终尺寸钻。[E1]、[E2]、[E3]、[E4] 的证据均支持这一顺序。中心钻用于初步定位，先导钻用于减少钻削力，最终尺寸钻用于精确钻孔。[E1]、[E2]、[E3]、[E4] 的证据均属于同一次记录。",
  "evidence_explanation": "[E1] 记录“drilling operation—Tool—final size drill”，对应原文 [D1]：“A drilling operation for through holes required center drilling first, followed by pilot drilling at 0.3 times the final diameter, and then finishing with the final size drill.” [E2] 记录“drilling operation—Tool—final diameter drill”，对应原文 [D1]：“A drilling operation for through holes required center drilling first, followed by pilot drilling at 0.3 times the final diameter, and then finishing with the final size drill.” [E3] 记录“drilling operation—Tool—pilot drill”，对应原文 [D1]：“A drilling operation for through holes required center drilling first, followed by pilot drilling at 0.3 times the final diameter, and then finishing with the final size drill.” [E4] 记录“drilling operation—Tool—center drill”，对应原文 [D1]：“A drilling operation for through holes required center drilling first, followed by pilot drilling at 0.3 times the final diameter, and then finishing with the final size drill.”",
  "cited_evidence": [
    "E1",
    "E2",
    "E3",
    "E4"
  ]
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work builds upon the [EDC (Extract-Define-Canonicalize)](https://github.com/clear-nus/edc) framework and extends it to the manufacturing domain with domain-specific prompt engineering, schema design, and Chain-of-Thought enhanced canonicalization.

## Citation

If you use MPKG in your research, please cite:

**APA:**
> Fu, Y., Liu, J., Guo, L., Liu, L., & Geng, X. (2026). An improved large language model and knowledge graph integration method for automated machining process base construction. *Journal of Manufacturing Systems*, *85*, 318–337. https://doi.org/10.1016/j.jmsy.2026.01.016

**BibTeX:**
```bibtex
@article{fu2026mpkg,
  title={An improved large language model and knowledge graph integration method for automated machining process base construction},
  author={Fu, Yan and Liu, Jie and Guo, Liang and Liu, Li and Geng, XiangYu},
  journal={Journal of Manufacturing Systems},
  volume={85},
  pages={318--337},
  year={2026},
  issn={0278-6125},
  doi={10.1016/j.jmsy.2026.01.016},
  publisher={Elsevier}
}
```
