# EyeAgent Benchmark 模块使用指南

我已经成功为 EyeAgent 的 multiagent 框架创建了一个完整的 benchmark 模块，用于评估诊断准确率。以下是使用指南：

## 📁 模块结构

```
eyeagent/benchmark/
├── __init__.py              # 模块初始化和导入
├── config.py                # 配置类定义
├── dataset_loader.py        # 数据集加载器
├── format_agent.py          # 输出格式化代理
├── metrics.py               # 评估指标计算
├── runner.py                # 主要运行器
├── cli.py                   # 命令行接口
├── README.md                # 详细文档
└── examples/                # 配置示例
    ├── README.md
    ├── basic_benchmark.yaml
    ├── dr_screening_benchmark.yaml
    ├── multi_disease_benchmark.yaml
    ├── dry_run_benchmark.yaml
    └── run_example.py
```

## 🚀 快速开始

### 1. 基本使用

```python
import asyncio
from eyeagent.benchmark import BenchmarkConfig, BenchmarkRunner
from eyeagent.benchmark.config import DatasetConfig

# 创建配置
config = BenchmarkConfig(
    dataset=DatasetConfig(
        name="eye_diseases",
        path="./data/classification_dataset.csv",
        image_column="image_path",
        label_column="diagnosis",
        class_names=["Normal", "DR", "AMD", "Glaucoma"]
    )
)

# 运行 benchmark
runner = BenchmarkRunner(config)
results = await runner.run_benchmark()

print(f"准确率: {results['metrics']['accuracy']:.3f}")
print(f"F1分数: {results['metrics']['f1_score']:.3f}")
```

### 2. 使用配置文件

```python
from eyeagent.benchmark import run_benchmark_from_config

# 从 YAML 配置文件运行
- 每个样本的 trace 会单独保存在 `cases/<case_id>/trace.json`，而 benchmark 还会在 `case_results/` 目录实时输出精简版结果，方便随时查看。

```bash
# 运行基本 benchmark
python eyeagent/benchmark/cli.py run --dataset ./data/test.csv --classes Normal DR AMD --output ./results

# 使用配置文件运行
python eyeagent/benchmark/cli.py run --config examples/basic_benchmark.yaml

# 创建示例数据集
python eyeagent/benchmark/cli.py create-dataset --output ./sample_data/test.csv --samples 20

# 生成配置模板
python eyeagent/benchmark/cli.py generate-config --output ./my_config.yaml --template basic
```

## 📊 核心功能

### 1. FormatAgent - 输出格式化

- 自动提取诊断结果
- 标准化输出格式：`"The diagnosis of this image is XXX"`
- 支持多种提取策略，确保鲁棒性
- 自动映射到有效的类别名称

### 2. 评估指标

支持的指标包括：
- **准确率 (Accuracy)**
- **精确率 (Precision)** - 宏平均/微平均/加权平均
- **召回率 (Recall)** - 宏平均/微平均/加权平均  
- **F1分数 (F1-Score)** - 宏平均/微平均/加权平均
- **AUC-ROC** - 二分类和多分类
- **混淆矩阵 (Confusion Matrix)**
- **每类别详细指标**

### 3. 数据集支持

支持多种格式：
- CSV 文件
- JSON 文件
- JSONL 文件
- 自定义眼科数据集格式

示例数据集格式：
```csv
image_path,diagnosis,patient_id
./images/001.jpg,Normal,P001
./images/002.jpg,DR,P002
./images/003.jpg,AMD,P003
```

### 预处理 VQA 诊断数据集

如果你使用的是 `datasets/VQA_diagnosis_20250801_*.csv` 系列文件，可以通过脚本一次性生成包含图片绝对路径的版本：

```bash
uv run python scripts/prepare_vqa_datasets.py
```

脚本会在 `datasets/processed/` 下生成三份文件：

- `VQA_diagnosis_20250801_OCT_with_paths.csv`
- `VQA_diagnosis_20250801_CFP_with_paths.csv`
- `VQA_diagnosis_20250801_SLO_with_paths.csv`

它们都包含统一的 `image_path` 列，方便直接用于 benchmark 配置。

## ⚙️ 配置示例

### 基本配置 (basic_benchmark.yaml)

```yaml
dataset:
  name: "basic_classification"
  path: "./data/classification_dataset.csv"
  image_column: "image_path"
  label_column: "diagnosis"
  class_names: ["Normal", "DR", "AMD", "Glaucoma"]
  max_samples: 100

model:
  workflow_backend: "langgraph"
  mcp_server_url: "http://localhost:8000/mcp/"
  dry_run: false
  enable_format_agent: true

metrics:
  compute_accuracy: true
  compute_auc: true
  compute_f1: true
  average: "macro"

output:
  output_dir: "./benchmark_results"
  save_predictions: true
  save_confusion_matrix: true
  verbose: true
```

## 📈 输出结果

Benchmark 会生成以下文件：

1. **benchmark_results_[timestamp].json** - 详细的每个样本结果
2. **metrics_report_[timestamp].json** - 完整的评估指标
3. **benchmark_config_[timestamp].yaml** - 使用的配置
4. **confusion_matrix.png** - 混淆矩阵可视化
5. **roc_curves.png** - ROC曲线（多分类）
6. **case_results/** - 每个样本单独的 JSON 结果（实时写入，便于调试）

示例输出结构：
```json
{
  "metrics": {

# 调试阶段只跑少量样本（例如前 5 条），并启用 dry-run
python eyeagent/benchmark/cli.py run --dataset ./data/test.csv --classes Normal DR AMD --max-samples 5 --dry-run
    "accuracy": 0.85,
    "f1_score": 0.82,
    "precision": 0.84,
    "recall": 0.81,
    "per_class": {
      "Normal": {"precision": 0.90, "recall": 0.88, "f1_score": 0.89},
      "DR": {"precision": 0.78, "recall": 0.75, "f1_score": 0.76}
    },
    "confusion_matrix": [[45, 2, 1, 0], [3, 38, 2, 1]]
  },
  "runtime": 125.3
}
```

## 🔧 高级功能

### 1. Dry-run 模式

用于测试管道而不实际运行模型推理：

```python
config.model.dry_run = True
```

### 2. 样本限制

限制处理的样本数量以加快测试：

```python
config.dataset.max_samples = 50
```

### 3. 自定义类别名称

指定有效的诊断类别：

```python
config.dataset.class_names = ["Normal", "Diabetic Retinopathy", "AMD", "Glaucoma"]
```

## 🛠️ 集成指南

### 与现有 EyeAgent 框架集成

Benchmark 模块完全兼容现有的 EyeAgent 架构：

- 使用相同的诊断工作流
- 兼容所有工作流后端（langgraph、profile、interaction）
- 支持现有的工具生态系统
- 可配置的 agent 设置
- 每个样本在 `cases/<case_id>/trace.json` 中保留全量 trace，同时 benchmark 会在 `case_results/` 下实时写出精简结果，随跑随看。

### 扩展 FormatAgent

FormatAgent 在现有 agent 流程后添加了一个格式化步骤：

```
Diagnostic Workflow → FormatAgent → Standardized Output → Evaluation
```

这确保了输出格式的一致性，便于评估。

## 📋 最佳实践

1. **数据集准备**
   - 确保图像路径正确
   - 使用一致的类别命名
   - 验证图像文件可读

2. **开发测试**
   - 先使用 dry-run 模式测试
   - 在开发期间使用较小的样本数量
   - 启用详细输出进行调试

3. **性能优化**
   - 对于大型数据集考虑批处理
   - 监控资源使用情况
   - 合理设置超时

## 🎯 使用场景

1. **模型性能评估** - 系统地评估诊断准确性
2. **A/B 测试** - 比较不同配置的性能
3. **持续集成** - 自动化模型质量检查
4. **研究评估** - 学术研究中的标准化评估

这个 benchmark 模块为 EyeAgent 提供了完整的评估框架，支持标准化的性能测试和指标计算，是确保诊断质量的重要工具。