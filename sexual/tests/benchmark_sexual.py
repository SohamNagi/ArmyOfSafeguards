"""
Basic benchmark evaluation for sexual content safeguard.

This script evaluates the fine-tuned model on the x_sensitive dataset.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import the safeguard
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from safeguard_sexual import predict
from tqdm import tqdm


def main():
    """Run basic benchmark evaluation."""
    print("\n" + "=" * 60)
    print("SEXUAL CONTENT BENCHMARK")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("cardiffnlp/x_sensitive", split="test")
    
    # Define label keys
    label_keys = ['conflictual', 'drugs', 'profanity', 'selfharm', 'sex', 'spam']
    
    # Collapse to binary labels (same as training)
    def collapse_to_binary(example):
        binary_label = 1 if any(example[key] == 1 for key in label_keys) else 0
        return {"true_label": binary_label}
    
    dataset = dataset.map(collapse_to_binary)
    
    # Limit to 500 samples for quick test
    dataset = dataset.select(range(min(500, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...\n")
    
    # Run predictions
    predictions = []
    true_labels = []
    
    for example in tqdm(dataset, desc="Processing"):
        text = example['text']
        true_label = example['true_label']
        
        result = predict(text)
        pred_label = 1 if result['label'] == 'LABEL_1' else 0
        
        predictions.append(pred_label)
        true_labels.append(true_label)
    
    # Calculate metrics
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    accuracy = correct / len(predictions)
    
    tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    tn = len(predictions) - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Negatives:  {tn}")
    print(f"{'='*60}\n")
    
    print("✅ Benchmark completed!")


if __name__ == "__main__":
    main()

