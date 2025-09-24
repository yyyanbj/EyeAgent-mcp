# Web Search Tools (web_search:pubmed, web_search:tavily)

Simple wrappers around PubMed e‑utilities and Tavily API.

## Tool IDs
- web_search:pubmed
- web_search:tavily

## Inputs / Outputs
- Input: `{ query: string, top_k?: number }`
- Output: `{ items: [...], source: "pubmed"|"tavily", inference_time: number, warning?: string }`

## Configuration

- PubMed 变体：不需要 API Key（小流量），网络失败或被墙时返回空 `items`。
- Tavily 变体：需要环境变量 `TAVILY_API_KEY`（若缺失将返回空 `items` 且包含 `warning` 字段）。

## 启动服务（MCP）

确保把 `eyetools/tools` 目录加入工具发现路径：

```bash
export EYETOOLS_TOOL_PATHS="/home/you/EyeAgent-mcp/eyetools/tools"
# 或
eyetools-mcp serve --tools-dir /home/you/EyeAgent-mcp/eyetools/tools
```

如果使用 Tavily：

```bash
export TAVILY_API_KEY="your_api_key_here"
```

## 直接调用（示例）

PubMed：
```json
POST /predict
{
	"tool_id": "web_search:pubmed",
	"request": { "inputs": { "query": "diabetic retinopathy treatment", "top_k": 3 } }
}
```

Tavily：
```json
POST /predict
{
	"tool_id": "web_search:tavily",
	"request": { "inputs": { "query": "retinal detachment latest guidelines", "top_k": 3 } }
}
```

## 常见问题（Troubleshooting）

- 工具未被发现：
	- 确认 `--tools-dir` 或 `EYETOOLS_TOOL_PATHS` 已包含 `eyetools/tools`。
	- 访问 `/tools` 或 `/mcp/tools` 确认 `web_search:pubmed`/`web_search:tavily` 是否存在。

- Tavily 空结果：
	- 检查 `TAVILY_API_KEY` 是否正确设置。
	- 网络受限或 API 响应异常时也会返回空列表。

- PubMed 空结果：
	- 可能是查询过于严格、网络被墙、或 PubMed 接口速率限制。
	- 换更简短或通用的关键词再试。

- 工具 ID 写法：
	- 注意是 `web_search:*`（带下划线），不是 `websearch:*`。

