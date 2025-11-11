# Sexual Content Safeguard Tests

This directory contains basic test scripts for the sexual content safeguard module.

## Test Files

### `quick_test.py`
Quick sanity check to verify the safeguard is working correctly.
- Tests basic safe and sensitive content detection
- Fast execution (~5 seconds)

**Usage:**
```bash
python sexual/tests/quick_test.py
```

### `test_sexual.py`
Basic unit tests for the sexual content safeguard.
- Basic prediction tests on safe and sensitive content
- Edge case handling (empty strings, long text)

**Usage:**
```bash
python sexual/tests/test_sexual.py
```

### `benchmark_sexual.py`
Basic benchmark evaluation on the x_sensitive dataset.
- Evaluates model performance on 500 test samples
- Calculates accuracy, precision, recall, F1-score
- Provides confusion matrix

**Usage:**
```bash
python sexual/tests/benchmark_sexual.py
```

### `evaluate_sexual.py`
Basic evaluation with metrics.
- Complete evaluation on full test set
- Saves results to JSON file

**Usage:**
```bash
python sexual/tests/evaluate_sexual.py
```

## Expected Performance

Based on training results:
- **Test Accuracy**: ~82.6%
- **Test F1-Score**: ~82.9%

## Running Tests

```bash
# Quick test (recommended)
python sexual/tests/quick_test.py

# Basic tests
python sexual/tests/test_sexual.py

# Benchmark (500 samples)
python sexual/tests/benchmark_sexual.py

# Full evaluation
python sexual/tests/evaluate_sexual.py
```

## Test Data

Tests use the `cardiffnlp/x_sensitive` dataset, which includes:
- Sexual content
- Profanity
- Self-harm references
- Drug-related content
- Conflictual content
- Spam

The dataset is automatically downloaded from Hugging Face when running the tests.

