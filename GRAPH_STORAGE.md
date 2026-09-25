# 从规范化三元组构建并保存领域知识图谱

`build_graph.py` 把 MPKG 已抽取、已规范化的三元组组织为有向属性图，并保存到一个 SQLite 文件。构图不加载大模型，也不重新执行 OIE、SD 或 SC。

## 与现有流水线的衔接

```text
原始文本 → OIE → SD → SC → 规范化三元组
                              ↓
               batch: records.jsonl + triples.jsonl
               run: result_at_each_stage.json
                              ↓
             build_graph.py → SQLite 图谱
```

批处理目录是推荐输入：`triples.jsonl` 提供三元组及原始行号，`records.jsonl` 提供原文和检查标记，`summary.json` 用于核对数量。某条文本即使因为其他三元组弃权而没有通过整行合规检查，其中已经规范化且属于目标模式的三元组仍会入图。`None` 所代表的弃权关系不会入图。

普通 `run.py` 输出可使用运行目录或 `iterN/result_at_each_stage.json`。运行目录自动选择最大的 `iterN`，避免把旧轮次与最终轮次重复入库。也可直接导入 `canon_kg.txt`；此文件不保存原文，因此该入口只能记录文件行号，无法提供文本证据。

## 图数据结构

| 表 | 作用 |
| --- | --- |
| `entities` | 实体节点，按去除首尾空格后的名称精确合并 |
| `relations` | 目标 CSV 中的关系名称和通用定义 |
| `edges` | 从主体指向客体的规范化关系；相同三元组只保留一条边 |
| `occurrences` | 该边在各原文中的每次出现，保留原始行号和三元组位置 |
| `documents` | 原文、处理状态和合规检查标记 |
| `datasets`、`schemas` | 输入批次及所用关系模式的来源和指纹 |

关系模式按 CSV 内容哈希隔离。同名关系若来自不同模式，不会误合并为同一种边。同一路径重复导入会在事务中替换该数据集的旧出处，其他数据集的共同边继续保留。可用 `--dataset-id` 指定一个稳定的数据集名称。

设计依据是 [EDC 的开源实现](https://github.com/clear-nus/edc)提供的“抽取—定义—规范化”结果，以及 [Neo4j 开源项目](https://github.com/neo4j/neo4j)使用的节点、有向关系和属性图模型。这里通过 Python 自带的 SQLite 实现可在本项目环境直接运行的本地持久化；唯一约束、外键和事务对应 [SQLite UPSERT](https://www.sqlite.org/lang_UPSERT.html)与[外键文档](https://www.sqlite.org/foreignkeys.html)中的数据完整性机制。这是本地属性图实现，不提供 Cypher 服务。

项目已有 `schemas/OWL/` 本体，但现有英文批处理使用独立的 `schemas/process_relations_en.csv`。脚本以每次抽取实际采用的 CSV 为准，不自行推测 CSV 关系与 OWL 属性的对应，也不推测实体类型。

## 运行

在项目根目录执行以下 PowerShell 命令：

```powershell
.venv\Scripts\python.exe -X utf8 build_graph.py ingest --input output\testprocess_full --schema schemas\process_relations_en.csv --db output\testprocess_graph_20260924.sqlite
.venv\Scripts\python.exe -X utf8 build_graph.py stats --db output\testprocess_graph_20260924.sqlite
.venv\Scripts\python.exe -X utf8 build_graph.py neighbors --db output\testprocess_graph_20260924.sqlite --entity "face milling" --limit 5
```

导入普通样例运行结果：

```powershell
.venv\Scripts\python.exe -X utf8 build_graph.py ingest --input output\example_cot_replay_20260924 --schema schemas\example_schema.csv --db output\example_graph_20260924.sqlite
```

批处理仍有未完成或失败的文本时，默认拒绝导入；需要先保存已完成部分时，可在 `ingest` 命令后添加 `--allow-partial`。关系不在所选目标模式中、三元组结构错误或报告计数不一致时会报错，避免无声丢弃。

## 验证

```powershell
.venv\Scripts\python.exe -X utf8 -m unittest evaluate.graph_build_checks -v
```

检查生成的数据库文件：

```powershell
.venv\Scripts\python.exe -X utf8 -c "import sqlite3; c=sqlite3.connect('output/testprocess_graph_20260924.sqlite'); print(c.execute('PRAGMA integrity_check').fetchone()[0]); print(c.execute('PRAGMA foreign_key_check').fetchall()); c.close()"
```

图中的边代表 MPKG 已规范化的抽取结果。入图校验保证结构、关系模式及出处一致；它不能证明每条边的语义正确性。实体目前只按名称精确合并，未做跨文本指代消解或同义实体合并。
