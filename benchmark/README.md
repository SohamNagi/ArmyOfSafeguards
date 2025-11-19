# Benchmarking Pipeline

This directory contains the benchmarking pipeline for evaluating the Army of Safeguards system against public safety benchmarks.

## Overview

The benchmarking pipeline (`run_benchmark.py`) evaluates the safeguarding system (via the aggregator) against public benchmark datasets for safety and moderation. It calculates standard metrics including accuracy, precision, recall, and F1-score.

## Supported Benchmarks

The pipeline supports the following public benchmarks:

1. **HarmBench** (`walledai/HarmBench`)
   - Jailbreak & harmful-content robustness benchmark
   - Tests the system's ability to detect harmful prompts

2. **JailbreakBench** (`JailbreakBench/JBB-Behaviors`)
   - Jailbreak attempt detection benchmark
   - Contains harmful and benign jailbreak prompts

3. **WildGuardMix** (`allenai/wildguardmix`)
   - Moderation / guardrail benchmark
   - Tests content moderation capabilities

## Usage

### Run a Single Benchmark

```bash
# Run HarmBench with 100 examples
python benchmark/run_benchmark.py --benchmark HarmBench --limit 100

# Run JailbreakBench with custom threshold
python benchmark/run_benchmark.py --benchmark JailbreakBench --threshold 0.8

# Run WildGuardMix with all examples
python benchmark/run_benchmark.py --benchmark WildGuardMix
```

### Run All Benchmarks

```bash
# Run all benchmarks with 100 examples each
python benchmark/run_benchmark.py --all --limit 100

# Run all benchmarks with custom threshold
python benchmark/run_benchmark.py --all --threshold 0.75
```

### Command-Line Options

- `--benchmark`: Name of the benchmark to run (HarmBench, JailbreakBench, or WildGuardMix)
- `--all`: Run all available benchmarks
- `--limit`: Maximum number of examples per benchmark (default: all)
- `--threshold`: Confidence threshold for flagging (default: 0.7)
- `--no-save`: Don't save results to JSON file

## Output

The pipeline provides:

1. **Console Output**: Real-time progress and summary metrics
2. **JSON Results**: Detailed results saved to timestamped JSON files

### Metrics Calculated

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **Per-Class Metrics**: Separate metrics for "safe" and "unsafe" classes

### Example Output

```
======================================================================
Evaluating: HarmBench
Jailbreak & harmful-content robustness benchmark
======================================================================
Loading dataset...
Loaded 100 examples
Running safeguarding system...

Evaluating: 100%|████████████████████| 100/100 [00:45<00:00,  2.22it/s]

======================================================================
Results for HarmBench:
======================================================================
  Total examples: 100
  Accuracy: 85.00%
  Precision: 82.35%
  Recall: 87.50%
  F1-Score: 84.85%
  Average Confidence: 0.78

  Ground Truth Distribution:
    Safe: 40 (40.0%)
    Unsafe: 60 (60.0%)

  Prediction Distribution:
    Safe: 35 (35.0%)
    Unsafe: 65 (65.0%)

  Confusion Matrix:
    Safe → Safe: 32
    Safe → Unsafe: 8 (False Positives)
    Unsafe → Safe: 3 (False Negatives)
    Unsafe → Unsafe: 57
```

## How It Works

1. **Dataset Loading**: Loads the specified benchmark dataset from HuggingFace
2. **Text Extraction**: Extracts text fields and ground truth labels
3. **Evaluation**: Runs each text through the aggregator's `evaluate_text()` function
4. **Metrics Calculation**: Compares predictions against ground truth labels
5. **Results**: Generates comprehensive metrics and saves to JSON

## Integration with Aggregator

The pipeline uses the aggregator (`aggregator/aggregator.py`) as the entry point:

- Calls `evaluate_text(text, threshold)` for each example
- Uses `is_safe` field to determine if content was flagged
- Aggregates results from all individual safeguards (factuality, toxicity, sexual, jailbreak)

## Adding New Benchmarks

To add a new benchmark:

1. Add configuration to `BENCHMARKS` dictionary in `run_benchmark.py`:
   ```python
   "NewBenchmark": {
       "hf_id": "dataset/name",
       "config": None,
       "split": "test",
       "text_field": "text",
       "label_field": "label",
       "label_mapping": {
           True: "safe",
           False: "unsafe"
       },
       "note": "Description of benchmark"
   }
   ```

2. Update the `load_benchmark_dataset()` function if special handling is needed

3. Test with: `python benchmark/run_benchmark.py --benchmark NewBenchmark --limit 10`

## Requirements

- Python 3.9+
- `datasets` library (for loading HuggingFace datasets)
- `scikit-learn` (for metrics calculation)
- `tqdm` (for progress bars)
- All dependencies from `requirements.txt`

## Notes

- The pipeline automatically handles different dataset structures
- Field names are matched case-insensitively for flexibility
- Results are saved with timestamps for tracking
- Large datasets can be limited using `--limit` for faster testing
- Threshold can be adjusted to balance precision/recall trade-offs

## Troubleshooting

If you encounter issues loading a dataset:

1. **Field Name Mismatches**: The pipeline tries to match field names case-insensitively. If a dataset uses different field names, update the `BENCHMARKS` configuration in `run_benchmark.py`.

2. **Dataset Structure Changes**: Some datasets may have different structures. Check the HuggingFace dataset page for the correct field names and update the configuration accordingly.

3. **Missing Labels**: If a dataset doesn't have labels in the expected format, the pipeline will skip those examples and show a warning.

4. **Memory Issues**: For very large datasets, use `--limit` to restrict the number of examples evaluated.

