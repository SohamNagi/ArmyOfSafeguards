# Army of Safeguards

**CS399 - UR2PhD Project** - A modular multiagent safeguarding system for LLM output detection

A modular collection of AI safety safeguards for detecting various types of harmful or problematic content.

## 🏗️ Project Structure

```
ArmyOfSafeguards/
├── factuality/              # Factuality checking safeguard
│   ├── safeguard_factuality.py
│   ├── README.md
│   └── tests/               # Factuality-specific tests
│       ├── test_factuality.py
│       ├── quick_test.py
│       ├── benchmark_factuality.py
│       ├── evaluate_factuality.py
│       └── EVALUATION_SUMMARY.md
├── toxicity/                # Toxicity detection
│   ├── safeguard_toxicity.py
│   ├── README.md
│   └── tests/               # Toxicity-specific tests
│       ├── test_toxicity.py
│       ├── quick_test.py
│       └── evaluate_toxicity.py
├── sexual/                  # Sexual content detection
│   ├── safeguard_sexual.py
│   ├── README.md
│   └── tests/               # Sexual content-specific tests
│       ├── test_sexual.py
│       ├── quick_test.py
│       └── evaluate_sexual.py
├── jailbreak/               # Jailbreak attempt detection
│   ├── safeguard_jailbreak.py
│   ├── README.md
│   └── tests/               # Jailbreak content-specific tests
│       ├── quick_test.py
│       └── benchmark_jailbreak_jbb.py
├── aggregator/              # Unified interface for all safeguards
│   ├── aggregator.py
│   └── README.md
├── requirements.txt         # Shared dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/SohamNagi/ArmyOfSafeguards.git
cd ArmyOfSafeguards

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Use Individual Safeguards

```bash
# Run factuality check
python factuality/safeguard_factuality.py "The Earth is flat."

# Run sexual content check
python sexual/safeguard_sexual.py "Your text to evaluate"

# Run toxicity check
python toxicity/safeguard_toxicity.py "Your text to evaluate"

# Run jailbreak check
python jailbreak/safeguard_jailbreak.py "Your text to evaluate"
```

### 3. Aggregator (All Safeguards)

The aggregator runs all available safeguards and provides a unified safety assessment:

```bash
# Run aggregator (includes factuality, sexual, toxicity, jailbreak)
python aggregator/aggregator.py "Your text to evaluate here"
```

## 📦 Safeguards Status

### ✅ Complete

#### Factuality Safeguard (Ajith)
- **Model**: `ajith-bondili/deberta-v3-factuality-small`
- **Purpose**: Detects factually incorrect or misleading statements
- **Performance**: 54-81% accuracy on out-of-distribution datasets
- **Documentation**: [factuality/README.md](factuality/README.md)
- **Tests**: [factuality/tests/README.md](factuality/tests/README.md)

#### Sexual Content Safeguard (Jian)
- **Model**: `faketut/x-sensitive-deberta-binary`
- **Purpose**: Detects sexual and sensitive content (profanity, self-harm, drugs, etc.)
- **Performance**: 82.6% accuracy, 82.9% F1-score on test set
- **Documentation**: [sexual/README.md](sexual/README.md)
- **Tests**: [sexual/tests/README.md](sexual/tests/README.md)

#### Toxicity Safeguard (Soham)
- **Model**: `SohamNagi/tiny-toxicity-classifier`
- **Purpose**: Detects toxic, racist, and hateful content
- **Performance**: 79% accuracy on ToxiGen test set
- **Documentation**: [toxicity/README.md](toxicity/README.md)
- **Tests**: [toxicity/tests/README.md](toxicity/tests/README.md)

#### Jailbreak Safeguard (Tommy)
- **Model**: `tommypang04/finetuned-model-jailbrak`
- **Purpose**: Detects jailbreak attempts in prompts
- **Documentation**: [jailbreak/README.md](jailbreak/README.md)

### 🚧 In Development
- Additional evaluation datasets and metrics

### ✅ Infrastructure Complete
- **Aggregator Framework**: Ready to integrate multiple safeguards
- **Testing Template**: Comprehensive test structure for teammates to follow
- **Documentation Template**: Clear pattern for documenting safeguards

## 🔧 Usage

### Individual Safeguards

**Factuality Safeguard**:
```python
from factuality.safeguard_factuality import predict

result = predict("The sky is blue.")
print(f"Label: {result['label']}, Confidence: {result['confidence']:.2%}")
```

**Sexual Content Safeguard**:
```python
from sexual.safeguard_sexual import predict

result = predict("This is a normal sentence.")
print(f"Label: {result['label']}, Confidence: {result['confidence']:.2%}")
```

**Toxicity Safeguard**:
```python
from toxicity.safeguard_toxicity import predict

result = predict("Hello, how are you?")
print(f"Label: {result['label']}, Confidence: {result['confidence']:.2%}")
```

**Jailbreak Safeguard**:
```python
from jailbreak.safeguard_jailbreak import predict

result = predict("Your prompt here")
print(f"Label: {result['label']}, Confidence: {result['confidence']:.2%}")
```

### Aggregator (All Safeguards)

**Python API**:
```python
from aggregator.aggregator import evaluate_text

# Runs all available safeguards (factuality, sexual, toxicity, jailbreak)
result = evaluate_text("Your text here", threshold=0.7)
print(f"Is Safe: {result['is_safe']}")
print(f"Individual Results: {result['individual_results']}")
```

**Command Line**:
```bash
python aggregator/aggregator.py "Text to check"
```

## 🧪 Testing & Evaluation

Each safeguard has its own test suite in its directory:

```bash
# Factuality tests
python factuality/tests/quick_test.py
python factuality/tests/test_factuality.py
python factuality/tests/evaluate_factuality.py

# Sexual content tests
python sexual/tests/quick_test.py
python sexual/tests/test_sexual.py
python sexual/tests/evaluate_sexual.py --limit 100

# Toxicity tests
python toxicity/tests/quick_test.py
python toxicity/tests/test_toxicity.py
python toxicity/tests/evaluate_toxicity.py --limit 100

# Jailbreak tests
python jailbreak/safeguard_jailbreak.py "Test prompt"
python jailbreak/tests/quick_test.py
python jailbreak/tests/benchmark_jailbreak.jbb.py
```

### Evaluation Results

**Factuality Safeguard Performance**:

⚠️ **Note**: Model trained on TruthfulQA & FEVER - use OOD datasets for true generalization.

**Out-of-Distribution (True Generalization)**:
| Dataset | Accuracy | F1-Score | Domain |
|---------|----------|----------|--------|
| VitaminC | 54.00% | 36.11% | General claims |
| Climate-FEVER | 81.00% | - | Climate-specific |
| LIAR | 81.00% | - | Political statements |

**Training Data (Sanity Check)**:
| Dataset | Accuracy | F1-Score |
|---------|----------|----------|
| FEVER | 84.00% | 78.38% |
| TruthfulQA | 75.00% | - |

**Sexual Content Safeguard Performance**:

⚠️ **Note**: Model trained on CardiffNLP x_sensitive dataset.

**Test Set Performance**:
| Metric | Score |
|--------|-------|
| Accuracy | 82.6% |
| F1-Score | 82.9% |

**Toxicity Safeguard Performance**:

⚠️ **Note**: Model trained on ToxiGen dataset.

**ToxiGen Test Set**:
| Metric | Score |
|--------|-------|
| Accuracy | 79.00% |
| Precision | 75.00% |
| Recall | 69.23% |
| F1-Score | 72.00% |

**Jailbreak Safeguard Performance**:

⚠️ **Note**: Model trained on TrustAIRLab/in-the-wild-jailbreak-prompts dataset.

**ToxiGen Test Set**:
| Metric | Score |
|--------|-------|
| Accuracy | 94.8248% |
| F1-Score | 65.7143% |

### Individual Safeguard Benchmark Datasets

- **Factuality**: TruthfulQA, FEVER, SciFact, VitaminC, Climate-FEVER
- **Sexual Content**: CardiffNLP x_sensitive
- **Toxicity**: ToxiGen, hate_speech18, civil_comments
- **Jailbreak**: JBB-Behaviors

See individual safeguard test directories for evaluation scripts.

### Safeguard System Benchmark Datasets
- **Jailbreak & harmful-content robustness**: [HarmBench](https://huggingface.co/datasets/walledai/HarmBench), [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
- **Moderation / guardrail benchmarks**: [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix)
- **Broader safety suites**: [HELM Safety](https://crfm.stanford.edu/helm/safety/latest/), have to check if it's opensource

## 🤝 Contributing

Each team member maintains their own safeguard module:

1. Create your safeguard in its own directory (e.g., `toxicity/`)
2. Implement `predict()` function that returns `{"label": str, "confidence": float}`
3. Add your safeguard to the aggregator
4. Include tests and documentation

## 📝 Requirements

- Python 3.9+
- PyTorch
- Transformers
- See `requirements.txt` for full list

## 📄 License

[Add license information]

## 👥 Team

- **Ajith**: Factuality Safeguard
- **Soham**: Toxicity Safeguard
- **Jian**: Sexual Content Safeguard
- **Tommy**: Jailbreak Safeguard
