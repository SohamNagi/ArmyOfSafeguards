"""Generate KNN reference data from Anthropic HH-RLHF dataset.

This script processes the HH-RLHF dataset and generates reference data
for the KNN aggregator by running all safeguards on the dataset.
"""
import json
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Add parent directory to path so we can import aggregator and safeguards
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import aggregator functions
from aggregator import run_all_safeguards

# 2. 下载 anthropic/hh-rlhf（只用 harmless 子集，标注最干净）
print("正在下载 anthropic/hh-rlhf (harmless 子集)...")
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")

# 3. 准备输出文件（保存在 aggregator 目录下）
output_file = Path(__file__).parent / "knn_reference_hh_rlhf_full.jsonl"
total = len(dataset)

print(f"开始处理 {total} 条样本，用你的 4 个模型预测中...")

with open(output_file, "w", encoding="utf-8") as f:
    for item in tqdm(dataset, total=total, desc="Predicting"):
        text = item["chosen"]  # chosen 是人类更喜欢的（通常是 safe 的）
        # 或者你也可以用 item["rejected"] 作为 unsafe 样本，这里我们用 chosen 为主

        # 关键：用你现有的 run_all_safeguards 拿到 4 个 confidence
        results = run_all_safeguards(text)

        # 提取 4 个 confidence（顺序必须和后面 KNN 一致！）
        # 顺序：factuality, toxicity, sexual, jailbreak
        conf_fact = results.get("factuality", {}).get("confidence", 0.0) if "error" not in results.get("factuality", {}) else 0.0
        conf_tox  = results.get("toxicity", {}).get("confidence", 0.0) if "error" not in results.get("toxicity", {}) else 0.0
        conf_sex  = results.get("sexual", {}).get("confidence", 0.0) if "error" not in results.get("sexual", {}) else 0.0
        conf_jb   = results.get("jailbreak", {}).get("confidence", 0.0) if "error" not in results.get("jailbreak", {}) else 0.0

        # harmless-base 的标签：chosen = safe, rejected = unsafe
        # 这里我们用 chosen 作为 safe 样本
        is_safe = True  # harmless-base 的 chosen 都是 safe

        record = {
            "text": text,
            "conf_fact": round(float(conf_fact), 4),
            "conf_tox":  round(float(conf_tox),  4),
            "conf_sex":  round(float(conf_sex),  4),
            "conf_jb":   round(float(conf_jb),   4),
            "is_safe": is_safe,
            "source": "hh-rlhf-harmless-chosen"
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"chosen 部分完成！已保存 {output_file}")

# ------------------- 可选：再加 rejected 部分（unsafe 样本）-------------------
print("正在处理 rejected 部分（unsafe 样本）...")
with open(output_file, "a", encoding="utf-8") as f:
    for item in tqdm(dataset, total=total, desc="Predicting rejected"):
        text = item["rejected"]  # rejected 是被拒绝的，通常 unsafe

        results = run_all_safeguards(text)
        # 顺序：factuality, toxicity, sexual, jailbreak
        conf_fact = results.get("factuality", {}).get("confidence", 0.0) if "error" not in results.get("factuality", {}) else 0.0
        conf_tox  = results.get("toxicity", {}).get("confidence", 0.0) if "error" not in results.get("toxicity", {}) else 0.0
        conf_sex  = results.get("sexual", {}).get("confidence", 0.0) if "error" not in results.get("sexual", {}) else 0.0
        conf_jb   = results.get("jailbreak", {}).get("confidence", 0.0) if "error" not in results.get("jailbreak", {}) else 0.0

        record = {
            "text": text,
            "conf_fact": round(float(conf_fact), 4),
            "conf_tox":  round(float(conf_tox),  4),
            "conf_sex":  round(float(conf_sex),  4),
            "conf_jb":   round(float(conf_jb),   4),
            "is_safe": False,  # rejected 通常是 unsafe
            "source": "hh-rlhf-harmless-rejected"
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"全部完成！最终参考集：{output_file}，共约 {total * 2} 条")

# ------------------- 自动 fit KNN（直接在脚本末尾）-------------------
print("正在 fit KNN（全量数据）...")
print("注意：KNN 参考数据已保存，使用 load_knn_reference_data() 函数来加载")
print(f"   参考数据文件: {output_file}")
print("   使用方法: from aggregator.aggregator import load_knn_reference_data")
print(f"            load_knn_reference_data('{output_file}')")