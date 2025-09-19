# 眼科多智能体诊断系统架构设计

## 系统目标
输入：患者基本资料（年龄、性别、主诉、既往史等）+ 多模态眼科影像（CFP、OCT、FFA ...）\n输出：结构化诊断报告（疾病结论、分级、病灶结果、随访/治疗建议）+ 完整 reasoning trace（可溯源）。

## Agent 角色
注：系统正逐步合并为“单一 Unified Agent（配置驱动）”架构，以下分角色描述用于阐释职责边界与推理视角，实际运行时由一个统一 Agent 依据 YAML 配置选择工具与决策路径。
| Agent | 功能 | 主要工具 | 关键输出 |
|-------|------|----------|----------|
| Orchestrator | 流程调度、模态识别、左右眼识别、任务规划 | 模态分类、左右眼分类 | 调用路径、决策 reasoning |
| Image Analysis | 影像质量、病灶检测、初步多疾病分类 | CFP 质量分类、病灶分割、疾病多分类 | 病灶结构、初步疾病列表、reasoning |
| Specialist | 专病确诊/分级、多模态量化 | 32 类专病分级模型、OCT/FFA 定量分析 | 各疾病分级、证据、reasoning |
| Follow-up | 风险/进展评估、管理策略 | 年龄分类、DR/AMD 分级结果 | 随访/治疗建议、reasoning |
| Report | 汇总所有中间结果并生成报告 | （无需外部工具） | Final Report + 全局 reasoning trace |

## Reasoning & Trace 机制
所有 Agent 输出统一包含：
```json
{
  "agent": "ImageAnalysisAgent",
  "role": "image_analysis",
  "inputs": {...},
  "tool_calls": [
    {"tool_id": "lesion_seg", "mcp_tool": "lesion_segmentation", "version": "1.0.0", "arguments": {...}, "output": {...}, "confidence": 0.92, "reasoning": "检测黄斑区出血"}
  ],
  "outputs": {"lesions": {...}, "diseases": [...]},
  "reasoning": "依据病灶特征与分类结果初步判断 DR 风险。"
}
```

## 诊断工作流（初版顺序，可扩展条件分支）
1. Orchestrator：识别模态 + 左右眼 → 规划后续调用（例如 CFP → ImageAnalysis → Specialist → FollowUp → Report）
2. Image Analysis：质量过滤、病灶提取、初步多疾病分类
3. Specialist：针对候选疾病调用专病模型（动态子集）
4. Follow-up：结合年龄 + 分级生成建议
5. Report：聚合所有步骤成最终结构化 JSON

未来可通过图结构（LangGraph / 自定义有向图）实现条件跳转、失败重试、替代工具回退。

## 工具注册 (Tool Registry)
集中维护工具元数据（ID、版本、描述、支持模态、所属阶段、输出 schema 简述），并在运行时与 MCP server 的实际工具名称映射。支持：
- 按角色/模态过滤
- 版本追踪
- Prompt 生成（用于 text grad 优化）

示例：
```python
TOOL_REGISTRY = {
  "modality_classify": {"mcp_name": "modality_classifier", "version": "1.0.0", "role": "orchestrator", "modalities": ["CFP","OCT","FFA"], "desc": "识别图像模态"},
  "dr_grading": {"mcp_name": "dr_grading_model", "version": "2.1.0", "role": "specialist", "disease": "DR", "desc": "糖尿病视网膜病变分级"}
}
```

## JSON 最终报告结构
```json
{
  "case_id": "uuid",
  "patient": {"patient_id": "P123", "age": 63, "gender": "M"},
  "images": [{"image_id": "IMG001", "path": "...", "modality": "CFP", "eye": "OD"}],
  "workflow": [ /* 每个 agent 的输出块，按时间顺序 */ ],
  "final_report": {
    "diagnoses": [{"disease": "DR", "grade": "R2", "confidence": 0.91, "evidence": ["microaneurysm", "hemorrhage"]}],
    "lesions": {"hemorrhage": 12, "exudate": 4},
    "management": {"follow_up_months": 3, "suggestion": "建议 3 个月后复查 OCT"},
    "reasoning": "基于多疾病分类与专病分级一致性..."
  },
  "trace_log_path": "<resolved path to trace.json>",
  "generated_at": "2025-09-17T12:00:00Z",
  "schema_version": "1.0.0"
}
```

## 错误溯源策略
- 每个工具调用事件记录：tool_id、mcp_tool、version、inputs、raw_output、parsed_output、confidence、latency、agent、reasoning
- 发生错误时（异常或低置信度），写入事件 `status = failed`，`error_type`、`stack`（可选）
- 溯源报告通过遍历 `workflow` + `tool_calls`，定位：
  - 调度错误：Orchestrator reasoning 与实际后续调用不一致
  - 病灶检测错误：lesion seg 输出缺失 / 低置信但仍被使用
  - 分级错误：专病分级置信度低或与初步分类冲突

## Prompt 优化准备
- Tool Registry 支持 `desc_long`，预留更丰富上下文
- Agent 基类生成 prompt 时动态拼接：当前病例上下文 + 已有中间结果摘要 + 工具列表 + 期望 JSON 输出 schema
- 预留 `prompt_variants/` 目录保存演化版本，用于 text grad 对比

## 目录结构（统一版，推荐）
```
eyeagent/
  agents/
    __init__.py
    base_agent.py
    diagnostic_base_agent.py      # 带 reasoning 与 trace hook
    registry.py                   # 配置驱动的 Agent 加载（默认统一 Agent）
    # 注：legacy 多 Agent 文件仍在仓库中，处于弃用状态，后续将移除
  config/
    settings.py
    prompts.py
    pipelines.py
    tools_description.py
  core/
    interaction_engine.py
    logging.py
  llm/
    json_client.py
    models.py
  tools/
    tool_registry.py              # 工具元数据 & 查询（支持按配置筛选）
    langchain_mcp_tools.py
  tracing/
    trace_logger.py               # 统一事件/报告持久化
  ui/
    app.py
  schemas/
    diagnosis_report_schema.json
  diagnostic_workflow.py          # 工作流调度入口（使用统一 Agent）
  run_diagnosis.py                # CLI 入口（优先使用该路径下的脚本）
```

提示：仓库根目录下的 `diagnostic_workflow.py` 与 `run_diagnosis.py` 为旧版重复入口，已弃用，后续清理；请使用 `eyeagent/` 目录下的对应脚本。

## 目录结构（legacy，待移除）
以下为早期多 Agent 形态下的文件，仅供历史参考：
```
eyeagent/
  agents/
    orchestrator_agent.py
    image_analysis_agent.py
    specialist_agent.py
    followup_agent.py
    report_agent.py
```

## 扩展与测试建议
- 单元测试：
  - mock MCP 工具返回 → 验证 agent reasoning 结构与输出 schema
  - TraceLogger：事件顺序、文件写入、失败标记
  - Orchestrator：输入不同模态组合 → 规划路径正确
- 集成测试：模拟一份 CFP + 患者资料 → 生成完整 report JSON

## 安全与合规
- 仅保存去标识化 patient_id
- 不在日志中写入原始影像像素（仅路径/ID）

## 下一步（可选）
- 引入配置驱动的 DAG（YAML）替代硬编码顺序
- 质量门控：自动检测低置信度链路并回退 / 触发复核
- LLM reasoning 裁剪与合并，避免泄露敏感数据
