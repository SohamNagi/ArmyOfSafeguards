# KNN Aggregator Workflow Script

## 快速开始

使用 `knn_workflow.py` 脚本可以一键完成整个 KNN aggregator 的工作流程。

### 基本用法

```bash
# 完整工作流（生成参考数据 + 评估性能）
python aggregator/knn_workflow.py
```

这个命令会：
1. 自动生成 KNN 参考数据（从 HH-RLHF 数据集）
2. 评估 KNN vs Majority Vote 的性能
3. 显示对比结果

### 常用选项

```bash
# 如果参考数据已存在，跳过生成步骤
python aggregator/knn_workflow.py --skip-generation

# 自定义评估样本数量
python aggregator/knn_workflow.py --limit 200

# 自定义 confidence 阈值
python aggregator/knn_workflow.py --threshold 0.8

# 只测试 KNN（不对比 Majority Vote）
python aggregator/knn_workflow.py --skip-generation --no-compare

# 强制重新生成参考数据
python aggregator/knn_workflow.py --force-generation
```

## 完整参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--skip-generation` | 跳过参考数据生成 | False |
| `--force-generation` | 强制重新生成参考数据 | False |
| `--reference-file` | 指定参考数据文件路径 | `aggregator/knn_reference_hh_rlhf_full.jsonl` |
| `--limit` | 评估样本数量 | 100 |
| `--threshold` | Confidence 阈值 | 0.7 |
| `--no-compare` | 不对比两种方法 | False |

## 工作流程示例

### 示例 1: 首次运行（完整流程）

```bash
python aggregator/knn_workflow.py --limit 100
```

输出：
```
============================================================
KNN Aggregator Workflow
============================================================
Configuration:
  Skip generation: False
  Force generation: False
  Evaluation limit: 100
  Threshold: 0.7
  Compare methods: True
============================================================

============================================================
Step 1: Generating KNN Reference Data from HH-RLHF
============================================================
正在下载 anthropic/hh-rlhf (harmless 子集)...
...

============================================================
Step 2: Evaluating Aggregator Performance (KNN vs Majority Vote Comparison)
============================================================
...
```

### 示例 2: 快速评估（跳过生成）

如果参考数据已经存在，可以跳过生成步骤：

```bash
python aggregator/knn_workflow.py --skip-generation --limit 200
```

### 示例 3: 测试不同阈值

```bash
python aggregator/knn_workflow.py --skip-generation --threshold 0.6 --limit 150
```

### 示例 4: 只测试 KNN（不对比）

```bash
python aggregator/knn_workflow.py --skip-generation --no-compare --limit 100
```

## 输出结果

脚本运行完成后会：

1. **生成参考数据文件**（如果运行了生成步骤）：
   - `aggregator/knn_reference_hh_rlhf_full.jsonl`

2. **生成评估结果文件**：
   - `aggregator/evaluation_results_compare_YYYYMMDD_HHMMSS.json`
   - 包含详细的性能指标和对比结果

3. **在终端显示**：
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - KNN vs Majority Vote 的性能提升

## 故障排除

### 问题：参考数据生成失败

**原因**：可能是网络问题或数据集下载失败

**解决**：
```bash
# 重试生成
python aggregator/knn_workflow.py --force-generation
```

### 问题：评估时找不到参考数据

**原因**：参考数据文件不存在或路径错误

**解决**：
```bash
# 指定正确的路径
python aggregator/knn_workflow.py --skip-generation --reference-file path/to/your/reference.jsonl
```

### 问题：导入错误

**原因**：Python 路径或依赖问题

**解决**：
```bash
# 确保在项目根目录运行
cd /path/to/ArmyOfSafeguards
python aggregator/knn_workflow.py
```

## 下一步

1. **调整 K 值**：编辑 `aggregator/aggregator.py` 中的 `knn_aggregator = KNNAggregator(k=7)`
2. **尝试不同的阈值**：使用 `--threshold` 参数测试
3. **使用不同的数据集**：修改 `evaluate_aggregator.py` 添加更多数据集支持

## 手动运行（分步执行）

如果你想分步执行，也可以手动运行：

```bash
# 步骤 1: 生成参考数据
python aggregator/generate_knn_reference_hh_rlhf_full.py

# 步骤 2: 评估性能
python aggregator/evaluate_aggregator.py \
    --dataset hh-rlhf \
    --limit 100 \
    --knn-reference aggregator/knn_reference_hh_rlhf_full.jsonl \
    --compare
```

## 相关文件

- `knn_workflow.py` - 本工作流脚本
- `generate_knn_reference_hh_rlhf_full.py` - 参考数据生成脚本
- `evaluate_aggregator.py` - 性能评估脚本
- `EVALUATION_GUIDE.md` - 详细评估指南

