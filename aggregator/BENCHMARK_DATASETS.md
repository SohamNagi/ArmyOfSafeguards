# Benchmark 数据集说明

## 当前使用的 Benchmark 数据集

### 1. HH-RLHF (Anthropic/hh-rlhf)

**数据集信息**:
- **HuggingFace ID**: `Anthropic/hh-rlhf`
- **子集**: `harmless-base`
- **描述**: Anthropic 的 Human Feedback 数据集，用于训练和评估 AI 安全系统
- **用途**: 测试模型对安全/不安全内容的识别能力

**数据结构**:
- `chosen`: 人类更偏好的回复（通常是安全的）
- `rejected`: 被拒绝的回复（通常是不安全的）

**标签映射**:
- `chosen` → `is_safe = True` (安全)
- `rejected` → `is_safe = False` (不安全)

**使用示例**:
```python
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
```

### 2. HarmBench (walledai/HarmBench)

**数据集信息**:
- **HuggingFace ID**: `walledai/HarmBench`
- **配置**: `standard` (还有 `contextual`, `copyright`)
- **描述**: Jailbreak 和有害内容鲁棒性 benchmark
- **用途**: 测试系统检测有害提示和 jailbreak 尝试的能力

**特点**:
- 所有样本都是有害的（red teaming dataset）
- 需要 HuggingFace 认证（gated dataset）

**使用示例**:
```python
from datasets import load_dataset
dataset = load_dataset("walledai/HarmBench", "standard", split="train")
```

### 3. WildGuardMix (allenai/wildguardmix)

**数据集信息**:
- **HuggingFace ID**: `allenai/wildguardmix`
- **配置**: `wildguardtest` (还有 `wildguardtrain`)
- **描述**: 内容审核/guardrail benchmark
- **用途**: 测试内容审核能力

**标签字段**:
- `prompt_harm_label`: `"unharmful"` 或 `"harmful"`

**使用示例**:
```python
from datasets import load_dataset
dataset = load_dataset("allenai/wildguardmix", "wildguardtest")
```

## 数据集对比

| 数据集 | 样本类型 | 标签来源 | 认证要求 | 主要用途 |
|--------|---------|---------|---------|---------|
| HH-RLHF | 对话回复 | chosen/rejected | 否 | 安全内容识别 |
| HarmBench | 有害提示 | 全部 unsafe | 是 | Jailbreak 检测 |
| WildGuardMix | 用户提示 | prompt_harm_label | 是 | 内容审核 |

## 在评估中使用

### 与 Majority Vote 对比

```bash
python aggregator/evaluate_aggregator.py \
    --dataset hh-rlhf \
    --limit 100 \
    --knn-reference aggregator/knn_reference_hh_rlhf_full.jsonl \
    --compare
```

### 与 IBM Granite 对比

```bash
python aggregator/evaluate_vs_granite.py \
    --dataset hh-rlhf \
    --limit 100 \
    --knn-reference aggregator/knn_reference_hh_rlhf_full.jsonl
```

## 添加新数据集

在 `evaluate_vs_granite.py` 或 `evaluate_aggregator.py` 中的 `BENCHMARKS` 字典添加：

```python
"your_dataset": {
    "dataset": "your/dataset-id",
    "split": "test",
    "text_field": "text",  # 文本字段名
    "label_field": "label",  # 标签字段名（可选）
    "description": "Your dataset description",
    "requires_auth": False,  # 是否需要认证
}
```

## 数据集访问

### HuggingFace 认证

对于需要认证的数据集（如 HarmBench）：

1. **获取 Token**:
   - 访问 https://huggingface.co/settings/tokens
   - 创建新 token（read 权限）

2. **接受条款**:
   - 访问数据集页面
   - 接受使用条款

3. **设置认证**:
   ```bash
   # 方法 1: 环境变量
   export HF_TOKEN=your_token_here
   
   # 方法 2: 命令行
   huggingface-cli login
   
   # 方法 3: Python
   from huggingface_hub import login
   login(token="your_token_here")
   ```

## 推荐使用

- **快速测试**: HH-RLHF（无需认证，数据量大）
- **Jailbreak 检测**: HarmBench（专门测试 jailbreak）
- **内容审核**: WildGuardMix（综合测试）

