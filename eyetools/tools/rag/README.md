# RAG Tool (rag:query)

Lightweight local RAG over markdown/txt sources. Uses simple token scoring to return top‑k snippets.

## Tool IDs
- rag:query

## Inputs / Outputs
- Input: `{ query: string, top_k?: number }`
- Output: `{ items: [{ title, text, source, score }], source: "rag", inference_time: number }`

## Configuration

RAG 会从一组语料目录中读取文本文件（.md/.txt/.rst/.py/.yml/.yaml），切分并建立轻量索引。

优先级（高→低）：
1. 环境变量 `EYETOOLS_RAG_DIRS`（使用冒号分隔多个路径，例如 `/data/docs:/opt/notes`）
2. 本包 `config.yaml` 中 `variants[].params.corpus_dirs`

注意：相对路径会相对工具包根目录（本目录）解析，因此无需依赖服务的当前工作目录（CWD）。

### 独立运行环境（不修改 eyetools 依赖）

本工具已配置为在独立环境下运行：`envs/py312-rag/pyproject.toml`，并且 `config.yaml` 中指定了：

- `shared.environment_ref: py312-rag`
- `runtime.load_mode: subprocess`

服务会通过 `uv run` 在子进程中按该环境的依赖列表（如 LangChain/Qdrant 等）执行工具，不会污染 `eyetools` 主项目依赖。

要求：系统可用的 Python 3.12 解释器（`python3.12`）。如果你的系统没有该解释器，请先安装，或在 `envs/py312-rag/pyproject.toml` 中调整 `requires-python`，并确保系统存在对应版本的解释器。

如需添加/固定 RAG 相关依赖，请修改 `envs/py312-rag/pyproject.toml` 的 `dependencies` 列表；无需修改 `eyetools/pyproject.toml`。

## 参数说明（config.yaml → variants[].params）

- mode: `qdrant` | `local`
	- qdrant：默认推荐，向量+稀疏混合检索（需要 warmup 进行向量化与入库）
	- local：轻量关键词匹配（无需额外依赖，功能较弱）
- top_k: 返回条数
- maxpages: 解析 PDF 的最大页数（控制 warmup 时长与存储量）
- collection_name: Qdrant 集合名
- vector_local_path: Qdrant 本地存储目录
- doc_local_path: 原始文本切片的本地存储目录
- chunk_size / chunk_overlap: 文本切片参数

## 预热（warmup）与检索

- warmup：在 `mode=qdrant` 时会执行以下步骤：
	1. 扫描 `corpus_dirs`（支持 .pdf/.md/.txt 等），按 `maxpages` 限制抽取 PDF 文本
	2. 依据 `chunk_size`/`chunk_overlap` 切片
	3. 使用 FastEmbed 生成向量，并写入本地 Qdrant；同时将切片保存至 `doc_local_path`
- predict：
	- `mode=qdrant` 使用混合相似度检索，直接返回切片内容与来源
	- `mode=local` 使用关键词匹配（BM25-like）

## 启动服务（MCP）

确保把 `eyetools/tools` 目录加入工具发现路径。例如：

```bash
# 任选一项（等价）：
export EYETOOLS_TOOL_PATHS="/home/you/EyeAgent-mcp/eyetools/tools"
# 或在启动时传参（示例 CLI）
eyetools-mcp serve --tools-dir /home/you/EyeAgent-mcp/eyetools/tools
```

可选：设置语料目录（支持 PDF）

```bash
export EYETOOLS_RAG_DIRS="/home/bingjie/workspace/EyeAgent-mcp/eyetools/weights/rag/books"

提示：本工具已支持基础 PDF 文本抽取（依赖 pdfminer.six 已在 rag 独立环境中安装）。大体积 PDF 会限制读取页数以避免过长的预热时间。
```

## 直接调用（示例）

HTTP（简化示例，具体以服务实际路由为准）：

POST /predict
```json
{
	"tool_id": "rag:query",
	"request": { "inputs": { "query": "retinal detachment signs", "top_k": 5 } }
}
```

返回：
```json
{
	"output": {
		"items": [ { "title": "...", "text": "...", "source": "/abs/path/file.md", "score": 1.23 } ],
		"source": "rag",
		"inference_time": 0.0123
	},
	"__meta__": { "tool_id": "rag:query", ... }
}
```

## 常见问题（Troubleshooting）

- 工具未被发现：
	- 检查是否设置了 `--tools-dir` 或 `EYETOOLS_TOOL_PATHS` 指向 `eyetools/tools`。
	- 访问 `/tools` 或 `/mcp/tools` 确认是否存在 `rag:query`。

- 调用成功但 items 为空：
	- 确认 `EYETOOLS_RAG_DIRS` 或 `config.yaml` 中的 `corpus_dirs` 是否指向真实存在的目录/文件。
	- 检查查询关键词是否确实出现在语料文件中（检索为简单关键词匹配，语义不匹配会无结果）。
	- 可先调用 `/admin/warmup?tool_id=rag:query` 触发预构建索引。

- 路由不一致（/ 与 /mcp）：
	- 设置 `EYETOOLS_MCP_MOUNT_PATH=/mcp` 以与客户端一致。
