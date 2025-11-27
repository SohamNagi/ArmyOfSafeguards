# KNN Aggregator Evaluation Guide

## 完整工作流程

### 步骤 1: 生成 KNN 参考数据

首先运行 `generate_knn_reference_hh_rlhf_full.py` 来生成参考数据集：

```bash
python aggregator/generate_knn_reference_hh_rlhf_full.py
```

这个脚本会：
1. 下载 Anthropic HH-RLHF 数据集（harmless-base 子集）
2. 对每个样本运行所有 4 个 safeguard（factuality, toxicity, sexual, jailbreak）
3. 提取每个模型的 confidence 值
4. 保存为 JSONL 格式：`aggregator/knn_reference_hh_rlhf_full.jsonl`
   - `chosen` 样本标记为 `is_safe: true`
   - `rejected` 样本标记为 `is_safe: false`

**输出文件格式**：
```json
{"text": "...", "conf_fact": 0.93, "conf_tox": 0.02, "conf_sex": 0.88, "conf_jb": 0.01, "is_safe": true}
```

### 步骤 2: 评估性能（比较 KNN vs Majority Vote）

运行评估脚本来测试性能提升：

```bash
# 比较 KNN 和 Majority Vote 的性能
python aggregator/evaluate_aggregator.py \
    --dataset hh-rlhf \
    --limit 100 \
    --knn-reference aggregator/knn_reference_hh_rlhf_full.jsonl \
    --compare
```

**参数说明**：
- `--dataset`: 要评估的数据集（目前支持 `hh-rlhf`）
- `--limit`: 评估的样本数量（默认 100）
- `--knn-reference`: KNN 参考数据文件路径
- `--compare`: 比较 KNN 和 Majority Vote 两种方法
- `--threshold`: Confidence 阈值（默认 0.7）

**只测试 KNN**：
```bash
python aggregator/evaluate_aggregator.py \
    --dataset hh-rlhf \
    --limit 100 \
    --knn-reference aggregator/knn_reference_hh_rlhf_full.jsonl
```

**只测试 Majority Vote**：
```bash
python aggregator/evaluate_aggregator.py \
    --dataset hh-rlhf \
    --limit 100 \
    --knn-reference "" \
    --no-compare
```

### 步骤 3: 查看结果

评估脚本会输出：
- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1 分数
- **Confusion Matrix**: 混淆矩阵
- **Improvement**: KNN 相比 Majority Vote 的提升（如果使用 `--compare`）

结果会自动保存为 JSON 文件：`evaluation_results_*.json`

## 评估指标说明

### 指标定义

- **Accuracy**: 正确预测的比例
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```

- **Precision**: 预测为 unsafe 的样本中，真正 unsafe 的比例
  ```
  Precision = TP / (TP + FP)
  ```

- **Recall**: 所有真正 unsafe 的样本中，被正确识别出的比例
  ```
  Recall = TP / (TP + FN)
  ```

- **F1-Score**: Precision 和 Recall 的调和平均数
  ```
  F1 = 2 * (Precision * Recall) / (Precision + Recall)
  ```

### 混淆矩阵

```
                Predicted
              Safe  Unsafe
Actual Safe    TP    FP
      Unsafe   FN    TN
```

- **TP (True Positive)**: 正确识别为 unsafe
- **FP (False Positive)**: 错误地将 safe 识别为 unsafe
- **FN (False Negative)**: 错误地将 unsafe 识别为 safe
- **TN (True Negative)**: 正确识别为 safe

## 预期结果

使用 KNN 聚合器相比 Majority Vote 的预期改进：

1. **更好的边界情况处理**: KNN 可以学习到复杂的决策边界
2. **减少 False Positives**: 通过参考相似样本，减少误报
3. **提高 Recall**: 更好地识别真正的 unsafe 内容

## 故障排除

### 问题：KNN 未加载

如果看到警告 "KNN not fitted yet!"，确保：
1. 已经运行了 `generate_knn_reference_hh_rlhf_full.py`
2. 在评估脚本中提供了正确的 `--knn-reference` 路径
3. 或者手动加载：
   ```python
   from aggregator.aggregator import load_knn_reference_data
   load_knn_reference_data("aggregator/knn_reference_hh_rlhf_full.jsonl")
   ```

### 问题：导入错误

确保在项目根目录运行脚本，或者正确设置了 Python 路径。

## 下一步

1. **调整 K 值**: 在 `aggregator.py` 中修改 `knn_aggregator = KNNAggregator(k=7)` 尝试不同的 k 值（5, 7, 9, 11）
2. **使用不同的参考数据集**: 可以生成其他数据集的参考数据
3. **调整阈值**: 使用 `--threshold` 参数测试不同的 confidence 阈值

