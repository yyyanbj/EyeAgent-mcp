# EyeAgent 工具规范化方案

## 概述

本文档描述了 EyeAgent 工具框架的规范化方案，实现了工具的标准化开发、打包、部署和环境隔离。

## 核心特性

### 1. 标准化工具结构
每个工具都是一个独立的 Python package，具有统一的结构：

```
tool_package/
├── __init__.py          # 包初始化和元数据
├── main.py              # Fire CLI 入口点
├── tool.py              # 工具实现类
├── config.yaml          # 工具配置和环境设置
├── requirements.txt     # Python 依赖
├── models/              # 模型文件（可选）
├── README.md            # 文档
└── tests/               # 单元测试（可选）
```

### 2. 统一接口
所有工具都继承自 `ToolBase` 类，实现标准方法：
- `execute()`: 执行工具
- `get_description()`: 获取描述
- `get_input_schema()`: 输入模式
- `get_output_schema()`: 输出模式
- `initialize()`: 初始化
- `load_model()`: 加载模型
- `preprocess()`: 预处理
- `inference()`: 推理
- `postprocess()`: 后处理

### 3. Fire CLI 接口
每个工具提供命令行接口：
```bash
# 初始化工具
python -m tool_package.main init

# 运行工具
python -m tool_package.main run --param value

# 获取描述
python -m tool_package.main describe

# 获取模式
python -m tool_package.main schema input
```

### 4. 环境隔离和复用
支持多种环境配置方式：

#### 直接 Python 可执行文件
```yaml
environment:
  python_executable: "/usr/bin/python3.9"
```

#### 虚拟环境路径
```yaml
environment:
  venv_path: "/path/to/venv"
```

#### Conda 环境
```yaml
environment:
  conda_env: "my_env"
```

#### 系统默认（最大复用）
```yaml
environment:
  python_executable: null
  venv_path: null
  conda_env: null
```

### 5. 自动依赖安装
工具部署时自动安装依赖：
- 读取 `requirements.txt`
- 使用指定的 Python 环境
- 支持版本约束

### 6. ZIP 打包部署
工具可以打包为 ZIP 文件：
```bash
zip -r tool_package.zip tool_package/
```

通过 Web UI 或 API 上传部署：
```bash
curl -X POST -F 'file=@tool_package.zip' http://localhost:8001/deploy
```

### 7. 多变体与集中配置 (v2)
v2 模板支持在单个 `config.yaml` 中声明多个变体（variants），并通过 `shared` 块复用通用字段，减少重复。常见于：
- 同一模型不同权重 (small / large / fp16)
- 不同类别集合但共享预处理
- 需要根据大小策略决定 `load_mode`（例如大模型走 subprocess）

统一新增字段（模板示例详见 `examples/tool_package_template/config.yaml` v2）：
| 字段 | 作用 |
| ---- | ---- |
| environment_ref | 指向预配置的基线环境 (envs/<ref>) 供 uv 运行 |
| runtime.device_policy | 设备策略：prefer_cuda / cpu_only / force_cuda / auto |
| runtime.idle_timeout_s | 空闲卸载时间窗 |
| model_defaults.* | 为所有变体提供默认模型加载策略 |
| lifecycle_hooks.* | 可选钩子，插入到 load/inference/unload 各阶段 |
| variants[].model.precision | 精度策略 (auto/fp16/bf16/int8) |
| variants[].runtime.load_mode | 可覆盖 shared.runtime.load_mode |

## 工具生命周期
标准生命周期（框架内部状态机）:

1. Discover (发现):
  - 解析 `config.yaml` 或 `config.py:get_config()`
  - 产生变体描述（未实例化模型）
2. Register (注册):
  - 写入注册表 (Registry)；可根据标签/类别过滤
3. Load (加载):
  - 触发条件：首次调用 / 预热 (preload) / 显式 `initialize()`
  - 解析权重路径 (glob) & 校验 (size/shape/classes)
  - 构建模型 & 放置设备 (device_policy)
  - 可执行 `before_load` / `after_load` 钩子
4. Execute (执行):
  - 预处理 preprocess()
  - 推理 inference()
  - 后处理 postprocess()
  - 包含 `before_inference` / `after_inference` 钩子
5. Idle Tracking (空闲跟踪):
  - 每次执行更新时间戳；后台监视器判断超过 idle_timeout_s
6. Unload (卸载):
  - 释放 GPU/内存 (del model, torch.cuda.empty_cache())
  - 调用 `before_unload` 钩子
7. Reload (按需复活):
  - 再次调用时自动进入 Load 阶段

生命周期状态示意：
DISCOVERED -> REGISTERED -> (LOADED <-> IDLE) -> UNLOADED -> LOADED ...

设计注意：
- 失败的 Load 应回退到 REGISTERED 状态，并缓存错误信息供诊断
- 执行中异常区分：可恢复 (预处理失败) vs 致命 (模型结构不匹配)
- 指标监控：load_time, last_exec_time, exec_latency_avg, failure_count


## 架构组件

### ToolBase 类
位于 `eyetools/base.py`，所有工具的基类：
- 提供通用功能
- 环境验证
- 执行流程管理

### ToolRegistry 类
位于 `eyetools/registry.py`：
- 工具注册和发现
- 从配置文件加载
- 从 ZIP 包加载

### ToolManager 类
位于 `eyetools/tool_manager.py`：
- 工具部署管理
- 环境信息查询
- 配置更新

### 主服务器
位于 `main_server.py`：
- FastAPI Web 服务
- MCP 集成
- 工具管理 API
- Web UI

## 使用流程

### 1. 创建工具
```bash
# 使用模板创建新工具
cp -r tool_template my_tool
cd my_tool

# 编辑配置文件
vim config.yaml

# 实现工具类
vim tool.py

# 添加依赖
vim requirements.txt
```

### 2. 测试工具
```bash
# 本地测试
python -m my_tool.main init
python -m my_tool.main run --image_path test.jpg
```

### 3. 打包工具
```bash
# 创建 ZIP 包
zip -r my_tool.zip my_tool/
```

### 4. 部署工具
```bash
# 启动服务器
python main_server.py

# 通过 Web UI 部署
# 访问 http://localhost:8001/ui
# 上传 my_tool.zip

# 或通过 API 部署
curl -X POST -F 'file=@my_tool.zip' http://localhost:8001/deploy
```

### 5. 使用工具
部署后，工具自动注册到 MCP，可以通过：
- MCP 协议调用
- 服务器 API
- Web UI

## 配置详解

### config.yaml 结构
```yaml
# 基本信息
name: "my_tool"
version: "1.0.0"
description: "My custom tool"
category: "classification"
author: "Your Name"

# 环境配置
environment:
  python_executable: null
  venv_path: null
  conda_env: null
  python_version: ">=3.8"
  dependencies:
    - "torch>=2.0.0"

# 工具参数
params:
  model_path: "models/model.pth"
  threshold: 0.5

# 输入输出模式
input_schema:
  type: "object"
  properties:
    image_path: {type: "string"}
  required: ["image_path"]

output_schema:
  type: "object"
  properties:
    predictions: {type: "array"}
```

## API 接口

### 工具管理 API
- `GET /list`: 列出所有工具
- `POST /enable/{tool_name}`: 启用工具
- `POST /disable/{tool_name}`: 禁用工具
- `POST /deploy`: 部署新工具
- `POST /undeploy/{tool_name}`: 卸载工具
- `GET /deployed`: 列出已部署工具
- `GET /environment/{tool_name}`: 获取环境信息
- `POST /validate/{tool_name}`: 验证环境
- `POST /update/{tool_name}`: 更新配置

### MCP 集成
所有工具自动注册为 MCP tools，支持：
- 工具发现
- 模式验证
- 统一调用接口

## 环境隔离策略

### 最大复用原则
- 默认使用系统 Python 环境
- 共享虚拟环境避免重复
- 按需创建隔离环境

### 环境检测
- Python 版本验证
- 依赖兼容性检查
- 路径有效性验证

### 依赖管理
- 自动安装缺失依赖
- 版本冲突检测
- 环境特定安装

## 最佳实践

### 工具开发
1. 遵循标准结构
2. 完整实现所有抽象方法
3. 提供详细的输入输出模式
4. 编写单元测试

### 环境配置
1. 优先使用系统环境
2. 为特殊需求指定专用环境
3. 明确依赖版本要求

### 部署管理
1. 定期验证工具环境
2. 监控资源使用
3. 及时清理不需要的工具

## 扩展性

### 添加新工具类型
1. 在 `ToolBase` 中添加新抽象方法
2. 更新 `registry.py` 中的类别映射
3. 修改模板文件

### 自定义环境类型
1. 扩展 `ToolBase.validate_environment()`
2. 更新 `ToolManager` 的环境处理
3. 添加新的环境配置选项

### 集成其他框架
1. 修改 `main.py` 的 CLI 接口
2. 更新服务器的协议支持
3. 扩展配置模式

## 总结

这个规范化方案提供了：
- **标准化**: 统一的工具开发和部署流程
- **灵活性**: 支持多种环境配置
- **可扩展性**: 易于添加新功能
- **易用性**: 简单的打包和部署过程
- **隔离性**: 环境隔离和依赖管理
- **复用性**: 最大程度复用现有环境

通过这个方案，可以高效地开发、部署和管理 EyeAgent 工具，实现工具生态的可持续发展。
