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

# Load Anthropic HH-RLHF dataset (harmless-base subset for clean labels)
print("Loading Anthropic/hh-rlhf dataset (harmless-base subset)...")
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")

# Prepare output file (saved in aggregator directory)
output_file = Path(__file__).parent / "knn_reference_hh_rlhf_full.jsonl"
total = len(dataset)

print(f"Processing {total} samples with 4 safeguard models...")

# Process chosen responses (typically safe)
with open(output_file, "w", encoding="utf-8") as f:
    for item in tqdm(dataset, total=total, desc="Processing chosen responses"):
        text = item["chosen"]  # chosen responses are preferred by humans (typically safe)
        
        # Run all safeguards to get 4 confidence values
        results = run_all_safeguards(text)

        # Extract 4 confidence values (order must match KNN aggregator!)
        # Order: factuality, toxicity, sexual, jailbreak
        conf_fact = results.get("factuality", {}).get("confidence", 0.0) if "error" not in results.get("factuality", {}) else 0.0
        conf_tox  = results.get("toxicity", {}).get("confidence", 0.0) if "error" not in results.get("toxicity", {}) else 0.0
        conf_sex  = results.get("sexual", {}).get("confidence", 0.0) if "error" not in results.get("sexual", {}) else 0.0
        conf_jb   = results.get("jailbreak", {}).get("confidence", 0.0) if "error" not in results.get("jailbreak", {}) else 0.0

        # Label mapping: chosen = safe, rejected = unsafe
        # Using chosen responses as safe samples
        is_safe = True  # harmless-base chosen responses are all safe

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

print(f"✅ Completed processing chosen responses. Saved to: {output_file}")

# Process rejected responses (unsafe samples)
print("Processing rejected responses (unsafe samples)...")
with open(output_file, "a", encoding="utf-8") as f:
    for item in tqdm(dataset, total=total, desc="Processing rejected responses"):
        text = item["rejected"]  # rejected responses are typically unsafe

        results = run_all_safeguards(text)
        # Order: factuality, toxicity, sexual, jailbreak
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
            "is_safe": False,  # rejected responses are typically unsafe
            "source": "hh-rlhf-harmless-rejected"
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Completed! Final reference dataset: {output_file}, approximately {total * 2} samples")

# Note: KNN reference data has been saved
print("\n" + "="*60)
print("KNN Reference Data Generation Complete")
print("="*60)
print(f"Reference data file: {output_file}")
print("\nTo load the reference data, use:")
print("  from aggregator.aggregator import load_knn_reference_data")
print(f"  load_knn_reference_data('{output_file}')")