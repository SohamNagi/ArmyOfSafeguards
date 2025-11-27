# 在 Google Colab 上运行 KNN Workflow

## 快速开始

### 方法 1: 使用 Jupyter Notebook（推荐）

1. **打开 Colab**：
   - 访问 [Google Colab](https://colab.research.google.com/)
   - 点击 "File" → "Upload notebook"
   - 上传 `aggregator/KNN_WORKFLOW_COLAB.ipynb`

2. **启用 GPU**（推荐）：
   - Runtime → Change runtime type → GPU
   - 点击 "Save"

3. **运行所有单元格**：
   - 按顺序运行每个单元格
   - 或使用 Runtime → Run all

### 方法 2: 使用 Python 脚本

1. **上传项目文件**：
   ```python
   # 在 Colab 中运行
   from google.colab import files
   uploaded = files.upload()  # 上传项目文件
   ```

2. **或者克隆仓库**：
   ```python
   !git clone https://github.com/SohamNagi/ArmyOfSafeguards.git
   %cd ArmyOfSafeguards
   ```

3. **安装依赖**：
   ```python
   !pip install -q transformers>=4.44 torch scikit-learn datasets==3.6.0 huggingface_hub safetensors tqdm pandas numpy
   ```

4. **运行工作流**：
   ```python
   !python aggregator/knn_workflow.py --limit 100
   ```

## 详细步骤

### 步骤 1: 准备 Colab 环境

```python
# 安装依赖
!pip install -q transformers>=4.44 torch scikit-learn datasets==3.6.0 huggingface_hub safetensors tqdm pandas numpy

# 克隆项目（或上传文件）
!git clone https://github.com/SohamNagi/ArmyOfSafeguards.git
%cd ArmyOfSafeguards
```

### 步骤 2: 启用 GPU（可选但推荐）

1. 点击菜单：**Runtime** → **Change runtime type**
2. 选择：**Hardware accelerator: GPU**
3. 点击：**Save**

### 步骤 3: 运行完整工作流

```python
# 方法 A: 使用工作流脚本（推荐）
!python aggregator/knn_workflow.py --limit 100

# 方法 B: 分步运行
# 1. 生成参考数据
!python aggregator/generate_knn_reference_hh_rlhf_full.py

# 2. 评估性能
!python aggregator/evaluate_aggregator.py \
    --dataset hh-rlhf \
    --limit 100 \
    --knn-reference aggregator/knn_reference_hh_rlhf_full.jsonl \
    --compare
```

### 步骤 4: 下载结果

```python
from google.colab import files

# 下载评估结果
files.download('aggregator/evaluation_results_compare_*.json')

# 可选：下载参考数据
# files.download('aggregator/knn_reference_hh_rlhf_full.jsonl')
```

## 使用 Google Drive 持久化存储

### 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 保存结果到 Drive

```python
# 复制结果到 Drive
!cp aggregator/evaluation_results_*.json /content/drive/MyDrive/
!cp aggregator/knn_reference_hh_rlhf_full.jsonl /content/drive/MyDrive/
```

### 从 Drive 加载已有数据

```python
# 如果参考数据已存在，跳过生成
!python aggregator/knn_workflow.py --skip-generation --limit 100
```

## 参数调整

### 快速测试（小数据集）

```python
!python aggregator/knn_workflow.py --limit 20
```

### 完整评估（中等数据集）

```python
!python aggregator/knn_workflow.py --limit 100
```

### 全面评估（大数据集）

```python
!python aggregator/knn_workflow.py --limit 500
```

### 自定义阈值

```python
!python aggregator/knn_workflow.py --limit 100 --threshold 0.8
```

## 常见问题

### Q: 如何检查 GPU 是否可用？

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Q: 运行时间太长？

- 减少 `--limit` 参数（如 `--limit 50`）
- 确保启用了 GPU
- 使用 `--skip-generation` 如果数据已存在

### Q: 内存不足？

- 减少 `--limit` 参数
- 重启运行时：Runtime → Restart runtime
- 使用较小的数据集

### Q: 如何保存进度？

- 使用 Google Drive 挂载
- 定期下载中间结果
- 使用 `--skip-generation` 跳过已完成的步骤

### Q: Colab 会话断开？

- 使用 Google Drive 保存结果
- 使用 `--skip-generation` 从断点继续
- 考虑使用 Colab Pro 获得更长会话时间

## 性能对比

| 环境 | 生成参考数据 | 评估 100 样本 |
|------|-------------|--------------|
| Colab (CPU) | 1-3 小时 | 10-20 分钟 |
| Colab (GPU) | 10-30 分钟 | 2-5 分钟 |
| 本地 (CPU) | 1-3 小时 | 10-20 分钟 |
| 本地 (GPU) | 10-30 分钟 | 2-5 分钟 |

## 完整示例代码

```python
# ============================================
# KNN Aggregator Workflow - Colab 完整示例
# ============================================

# 1. 安装依赖
!pip install -q transformers>=4.44 torch scikit-learn datasets==3.6.0 huggingface_hub safetensors tqdm pandas numpy

# 2. 克隆项目
!git clone https://github.com/SohamNagi/ArmyOfSafeguards.git
%cd ArmyOfSafeguards

# 3. 检查 GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")

# 4. 运行完整工作流
!python aggregator/knn_workflow.py --limit 100

# 5. 查看结果
import json
from pathlib import Path
results = list(Path("aggregator").glob("evaluation_results_*.json"))
if results:
    with open(max(results, key=lambda p: p.stat().st_mtime)) as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))

# 6. 下载结果
from google.colab import files
if results:
    files.download(str(max(results, key=lambda p: p.stat().st_mtime)))
```

## 总结

✅ **Colab 优势**：
- 免费 GPU 加速
- 无需本地安装
- 易于分享和协作

⚠️ **注意事项**：
- 会话有时间限制（免费版 ~12 小时）
- 需要网络连接
- 数据存储在云端

📝 **推荐流程**：
1. 使用 GPU 运行时
2. 挂载 Google Drive 保存结果
3. 使用 `--skip-generation` 从断点继续
4. 定期下载重要结果

