# Sexual Content Safeguard

**Author**: Jian

A safeguard module for detecting sexual and sensitive content in text using a fine-tuned DeBERTa-v3 model.

## Overview

This safeguard detects sexual content and other sensitive material (including profanity, self-harm, drugs, conflictual content, and spam) in text. The model was trained on the `cardiffnlp/x_sensitive` dataset and performs binary classification to identify sensitive content.

## Model Information

- **Model**: `faketut/x-sensitive-deberta-binary`
- **Base Model**: `microsoft/deberta-v3-base`
- **Task**: Binary text classification (sensitive vs. not sensitive)
- **Labels**:
  - `LABEL_0`: Not sensitive (safe content)
  - `LABEL_1`: Sensitive/sexual content (unsafe)

## Performance

- **Test Accuracy**: 82.6%
- **Test F1-Score**: 82.9%
- **Dataset**: CardiffNLP x_sensitive dataset (binary classification)

## Installation

The safeguard requires the following dependencies (already in `requirements.txt`):
- `torch`
- `transformers`
- `huggingface_hub`

## Usage

### Python API

```python
from sexual.safeguard_sexual import predict

# Single prediction
result = predict("This is a normal sentence.")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")

# Example with sensitive content
result = predict("This is so fucking cool!")
print(f"Label: {result['label']}")  # LABEL_1
print(f"Confidence: {result['confidence']:.2%}")
```

### Command Line

```bash
# Run sexual content check
python sexual/safeguard_sexual.py "Text to check"

# Interactive mode (if no text provided)
python sexual/safeguard_sexual.py
```

### Aggregation

```python
from sexual.safeguard_sexual import predict, aggregate

# Multiple predictions with aggregation
predictions = [
    predict("This is a normal sentence."),
    predict("Another safe text."),
    predict("Sensitive content here."),
]
result = aggregate(predictions)
print(f"Aggregated: {result['label']} ({result['votes']}/{result['total']} votes)")
```

## Integration with Aggregator

The sexual content safeguard is integrated into the main aggregator:

```python
from aggregator.aggregator import evaluate_text

result = evaluate_text("Your text here", threshold=0.7)
print(f"Is Safe: {result['is_safe']}")
print(f"Individual Results: {result['individual_results']}")
```

## Testing

Run the test suite:

```bash
# Unit tests
python sexual/tests/test_sexual.py

# Quick sanity check
python sexual/tests/quick_test.py

# Benchmark evaluation
python sexual/tests/benchmark_sexual.py

# Full evaluation with metrics
python sexual/tests/evaluate_sexual.py
```

## Model Training

The model was trained using:
- **Dataset**: `cardiffnlp/x_sensitive` (collapsed to binary labels)
- **Training epochs**: 3
- **Learning rate**: 2e-5
- **Batch size**: 16
- **Max length**: 128 tokens
- **Optimization**: Full fine-tuning on DeBERTa-v3-base

See `sexual_finetune_v1.ipynb` and `sexual_finetune_v2.ipynb` for training details.

## Label Interpretation

- **LABEL_0**: Content is safe (not sensitive)
- **LABEL_1**: Content contains sexual or sensitive material (unsafe)

The model detects various types of sensitive content including:
- Sexual content
- Profanity
- Self-harm references
- Drug-related content
- Conflictual content
- Spam

## Notes

- The model uses a max length of 128 tokens for efficiency
- Confidence scores are normalized probabilities from the softmax output
- The model is loaded once and cached for efficient inference

