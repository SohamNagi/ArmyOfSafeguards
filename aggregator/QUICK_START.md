# KNN Aggregator 快速开始

## 一键运行完整工作流

```bash
python aggregator/knn_workflow.py
```

这个命令会自动完成：
1. ✅ 生成 KNN 参考数据（从 HH-RLHF 数据集）
2. ✅ 评估 KNN vs Majority Vote 性能
3. ✅ 显示对比结果和性能提升

## 常用命令

```bash
# 完整工作流（首次运行）
python aggregator/knn_workflow.py

# 快速评估（跳过数据生成，如果数据已存在）
python aggregator/knn_workflow.py --skip-generation

# 自定义参数
python aggregator/knn_workflow.py --limit 200 --threshold 0.8

# 只测试 KNN（不对比）
python aggregator/knn_workflow.py --skip-generation --no-compare
```

## 输出文件

- **参考数据**: `aggregator/knn_reference_hh_rlhf_full.jsonl`
- **评估结果**: `aggregator/evaluation_results_compare_*.json`

## 查看帮助

```bash
python aggregator/knn_workflow.py --help
```

## 详细文档

- [KNN_WORKFLOW_README.md](KNN_WORKFLOW_README.md) - 完整使用指南
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - 评估指标说明

