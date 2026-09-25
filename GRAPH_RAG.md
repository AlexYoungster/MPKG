# MPKG 图谱问答后端

这是一个可在当前项目环境直接运行的最小 Graph RAG 后端。它使用已保存的 SQLite 领域图谱，默认由本地 **Qwen3-1.7B** 生成面向加工技术人员的中文回答，并随回答返回可追溯的图谱事实和原文出处。HTTP 服务仅使用 Python 标准库，不需要额外部署 Web 框架或图数据库服务。

## 数据流与设计边界

```text
问题 → 初始图谱定位 → Qwen Agent 判断证据是否充分
                         ↓不足
                  选择只读图谱工具并再次查询（可多轮、多跳）
                         ↓充分
        → 同一原文范围内合并证据 → Qwen 组织中文技术段落
        → answer + evidence_explanation + evidence(JSON)
```

这一实现采用 [Microsoft GraphRAG 开源工作的局部图谱检索思想](https://github.com/microsoft/graphrag/blob/main/docs/index.md)，但复用 MPKG 已建成的图谱，不重新做社群摘要或图索引。模型调用遵循 [Qwen3-1.7B 模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)的 Transformers 聊天模板和 `enable_thinking=False` 用法，并沿用项目已有的 8-bit 装载方式以适配本机 4 GB 显卡。

图中“车削”等工序实体可能连接多个工件。检索时，两跳关系只在**同一数据集、同一原文行**内组合；如果同一个实体仍对应多条候选原文，且问题无法唯一定位记录，后端要求补充工件或来源信息。没有明确命中的实体时，不凭向量相似度给出工艺参数。原文与机器抽取的图谱事实同时交给 Qwen，提示原文优先。

### 多阶段 Agent 查询

当生成器提供 `agent_step`（当前 `QwenGenerator` 默认提供）时，`GraphRAG` 会运行一个有界的工具循环。模型对外只看到一个 `graph_query` 接口，并在 JSON 中自行构造查询表达式：可以选择 `aggregate`、`match`、`path/traverse`、`union`，以及过滤、排序、分页等字段。程序只解释和执行参数化的只读查询，限制最多 6 轮、每轮最多 20 条原文或 200 个聚合值，并在每轮后重新合并证据。关系名称及其图谱定义会交给模型，由模型自行决定先定位工件、沿关系多跳，还是建立全库聚合目录；程序不为“全库概览”“比较”“分类”等问题增加专用工具。

每个响应的 `retrieval` 字段包含 `agent_status`、`agent_assessment`、`agent_rounds` 和 `agent_trace`，用于审计模型生成的查询表达式及其充分性判断。状态为 `sufficient` 表示模型主动判断当前证据足够；`insufficient` 表示模型确认图谱无法回答；`step_limit` 表示达到轮数上限而没有完成核验。候选记录或聚合值被截断时，后端把这一事实交给模型，由模型决定是否继续分页；不会用固定规则覆盖模型的充分性结论。聚合结果放在 `evidence.catalogs` 中，使用 `[C1]` 等编号引用，引用对应的完整原文会写入 `evidence_explanation`。

若“材料、刀具、冷却介质”等类别关系的客体却是带单位的量值，后端将其标记为类型冲突，不向 Qwen 提供为可用的类别事实。若生成回答仍复用了该量值，后端返回无法通过证据校验的说明、原文行号和 `grounding_warning`，避免把流量等参数写成介质名称。

## 启动

若尚未构图，先在项目根目录运行：

```powershell
.venv\Scripts\python.exe -X utf8 build_graph.py ingest --input output\testprocess_full --schema schemas\process_relations_en.csv --db output\testprocess_graph_20260924.sqlite
```

启动英文全量数据的问答服务：

```powershell
.venv\Scripts\python.exe -X utf8 qa_backend.py serve --db output\testprocess_graph_20260924.sqlite --offline --host 127.0.0.1 --port 8765
```

服务会在首个需要生成回答的请求中加载 Qwen，随后复用同一个模型实例。构图文件更新后，重启服务以重新读取图谱。服务默认只监听本机地址。

`--max-agent-steps` 可限制模型自主查询轮数，默认 6。更大的值允许跨更多文档分页和多跳追踪，但会增加模型调用时间；达到上限时响应会标记 `retrieval.agent_status=step_limit`，由于模型没有提交 finish 核验结论，后端不会生成完整答案。

另一终端可用 PowerShell 发送问题：

```powershell
$payload = @{ question = '钢轴车削时的主轴转速是多少？' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8765/ask' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

提供精确实体名称可用于消歧：

```powershell
$payload = @{ question = '主轴转速是多少？'; entity = 'steel shaft' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8765/ask' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

`GET /health` 返回服务状态、图谱路径和模型是否已加载。`POST /ask` 接受 `question` 和可选的 `entity`。响应包含自然语言 `answer`、`retrieval` 定位信息、`evidence_explanation`，以及 `evidence.documents` 原文和 `evidence.facts` 三元组；`[D1]` 和 `[E1]` 是回答与证据之间的对应编号。

不启动服务也可直接调试：

```powershell
.venv\Scripts\python.exe -X utf8 qa_backend.py ask --db output\testprocess_graph_20260924.sqlite --question '钢轴车削时的主轴转速是多少？' --offline
```

中文样例图谱可将 `--db` 改为 `output\example_graph_20260924.sqlite`，例如提问“零件T-6A1-4V的材料、加工工序和主轴转速分别是什么？”。

## 测试

```powershell
.venv\Scripts\python.exe -X utf8 -m unittest evaluate.graph_build_checks evaluate.graph_rag_checks -v
```

测试包含同名工序跨工件防串用、实体未命中时停止生成、中文检索句转换、出处编号和 HTTP 接口。真实模型测试可使用上面的 `ask` 命令。图谱及原文来自自动抽取，回答的可靠性仍受上游抽取质量影响；证据字段用于逐条人工核查。
