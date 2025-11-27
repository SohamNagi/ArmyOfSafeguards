# 本地运行指南

## ✅ 可以在本地运行，不需要 Colab！

所有脚本都可以在本地运行，只需要：
1. Python 3.9+
2. 安装依赖包
3. 网络连接（用于下载模型和数据集）

## 系统要求

### 最低要求
- **Python**: 3.9 或更高版本
- **内存**: 至少 8GB RAM（推荐 16GB+）
- **存储**: 至少 10GB 可用空间（用于模型缓存）
- **网络**: 稳定的互联网连接（首次运行需要下载模型）

### GPU（可选）
- **CPU 运行**: ✅ 完全支持，但速度较慢
- **GPU 运行**: ✅ 推荐（如果有 NVIDIA GPU），速度更快
- **无 GPU**: 也可以运行，只是推理速度会慢一些

## 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
# 测试单个 safeguard
python sexual/safeguard_sexual.py "Test text"

# 测试 aggregator
python aggregator/aggregator.py "Test text"
```

## 运行 KNN 工作流

### 完整工作流（首次运行）

```bash
python aggregator/knn_workflow.py
```

**首次运行会：**
1. 自动下载 Hugging Face 模型（约 500MB-2GB，取决于模型）
2. 下载 HH-RLHF 数据集（约 100-500MB）
3. 运行所有 4 个 safeguard 生成参考数据
4. 评估性能

**预计时间：**
- CPU: 1-3 小时（取决于数据集大小）
- GPU: 10-30 分钟

### 快速评估（如果数据已存在）

```bash
python aggregator/knn_workflow.py --skip-generation --limit 50
```

## 模型下载位置

模型会自动下载并缓存到：
- **Windows**: `C:\Users\<username>\.cache\huggingface\`
- **Linux/Mac**: `~/.cache/huggingface/`

首次运行后，模型会缓存，后续运行会更快。

## 性能优化建议

### 如果运行很慢

1. **减少评估样本数量**：
   ```bash
   python aggregator/knn_workflow.py --limit 50
   ```

2. **使用 GPU（如果有）**：
   - 安装 CUDA 版本的 PyTorch
   - 模型会自动使用 GPU（如果可用）

3. **分批处理**：
   - 先生成参考数据：`python aggregator/generate_knn_reference_hh_rlhf_full.py`
   - 再评估：`python aggregator/knn_workflow.py --skip-generation`

### 检查 GPU 是否可用

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## 常见问题

### Q: 需要 Colab 吗？
**A: 不需要！** 所有代码都可以在本地运行。

### Q: 需要 GPU 吗？
**A: 不需要，但推荐。** CPU 也可以运行，只是速度较慢。

### Q: 首次运行很慢？
**A: 正常。** 首次运行需要下载模型和数据集，后续会快很多。

### Q: 内存不足？
**A: 可以：**
- 减少 `--limit` 参数
- 关闭其他程序
- 使用更小的数据集

### Q: 网络问题？
**A: 可以：**
- 使用 VPN 或代理
- 手动下载模型到缓存目录
- 使用镜像站点

## 与 Colab 对比

| 特性 | 本地运行 | Colab |
|------|---------|-------|
| **需要网络** | ✅ 首次下载后离线可用 | ✅ 需要持续连接 |
| **GPU 支持** | ✅ 如果有 NVIDIA GPU | ✅ 免费 GPU（有限时） |
| **数据持久化** | ✅ 本地存储 | ⚠️ 需要挂载 Google Drive |
| **速度** | 取决于硬件 | 取决于 Colab 资源 |
| **隐私** | ✅ 完全本地 | ⚠️ 数据在云端 |
| **成本** | ✅ 免费 | ✅ 免费（有限制） |

## 推荐配置

### 快速测试（小数据集）
```bash
python aggregator/knn_workflow.py --limit 20
```

### 完整评估（中等数据集）
```bash
python aggregator/knn_workflow.py --limit 100
```

### 全面评估（大数据集）
```bash
python aggregator/knn_workflow.py --limit 500
```

## 总结

✅ **完全可以在本地运行**  
✅ **不需要 Colab**  
✅ **CPU 也可以运行**（只是慢一些）  
✅ **首次运行需要下载模型**（后续会缓存）  
✅ **推荐使用 GPU**（如果有，会更快）

