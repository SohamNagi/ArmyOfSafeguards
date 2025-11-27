# KNN Aggregator vs IBM Granite-4.0-H-Tiny 对比

## 工作流程

### 步骤 1: 采样生成 KNN 参考数据

运行 `generate_knn_reference_hh_rlhf_full.py` 从 HH-RLHF 数据集采样并生成参考数据：

```bash
python aggregator/generate_knn_reference_hh_rlhf_full.py
```

这会：
- 下载 HH-RLHF 数据集（harmless-base 子集）
- 对每个样本运行所有 4 个 safeguard（factuality, toxicity, sexual, jailbreak）
- 提取 confidence 值
- 保存为 `knn_reference_hh_rlhf_full.jsonl`

### 步骤 2: 并行评估并对比

运行 `evaluate_vs_granite.py` 在 benchmark 数据集上并行评估两个模型：

```bash
python aggregator/evaluate_vs_granite.py \
    --dataset hh-rlhf \
    --limit 100 \
    --knn-reference aggregator/knn_reference_hh_rlhf_full.jsonl
```

这会：
1. 加载 KNN 参考数据
2. 加载 IBM Granite-4.0-H-Tiny 模型
3. 在相同的 benchmark 数据集上并行评估两个模型
4. 计算并显示性能对比（Accuracy, Precision, Recall, F1-Score）

## 一键运行（完整工作流）

```bash
python aggregator/knn_workflow.py
```

这会自动执行：
1. 生成 KNN 参考数据（如果不存在）
2. 对比 KNN Aggregator vs IBM Granite

## Benchmark 数据集

当前支持：
- **hh-rlhf**: Anthropic HH-RLHF harmless-base（默认）
- **harmbench**: HarmBench（需要认证）
- **wildguardmix**: WildGuardMix（需要认证）

## 输出结果

脚本会生成 JSON 结果文件，包含：
- KNN Aggregator 的性能指标
- IBM Granite 的性能指标
- 性能改进对比（KNN vs Granite）

## 文件说明

- `generate_knn_reference_hh_rlhf_full.py` - 采样生成参考数据
- `evaluate_vs_granite.py` - 并行评估并对比两个模型
- `knn_workflow.py` - 完整工作流脚本

