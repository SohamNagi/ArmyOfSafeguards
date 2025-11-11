"""
Basic evaluation script for sexual content safeguard.

This script provides evaluation metrics on the test set.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import the safeguard
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from safeguard_sexual import predict
from tqdm import tqdm
import json


def main():
    """Run basic evaluation."""
    print("\n" + "=" * 60)
    print("SEXUAL CONTENT EVALUATION")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("cardiffnlp/x_sensitive", split="test")
    
    # Define label keys
    label_keys = ['conflictual', 'drugs', 'profanity', 'selfharm', 'sex', 'spam']
    
    # Collapse to binary labels
    def collapse_to_binary(example):
        binary_label = 1 if any(example[key] == 1 for key in label_keys) else 0
        return {"true_label": binary_label}
    
    dataset = dataset.map(collapse_to_binary)
    
    print(f"Evaluating on {len(dataset)} samples...\n")
    
    # Run predictions
    predictions = []
    true_labels = []
    confidences = []
    
    for example in tqdm(dataset, desc="Processing"):
        text = example['text']
        true_label = example['true_label']
        
        result = predict(text)
        pred_label = 1 if result['label'] == 'LABEL_1' else 0
        
        predictions.append(pred_label)
        true_labels.append(true_label)
        confidences.append(result['confidence'])
    
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
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    avg_confidence = sum(confidences) / len(confidences)
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nMetrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:    {f1:.4f} ({f1*100:.2f}%)")
    print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp:5d}")
    print(f"  False Positives: {fp:5d}")
    print(f"  False Negatives: {fn:5d}")
    print(f"  True Negatives:  {tn:5d}")
    
    print(f"\nConfidence:")
    print(f"  Average: {avg_confidence:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    results = {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity
        },
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        },
        "confidence": {
            "average": avg_confidence
        }
    }
    
    with open("evaluation_sexual.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to evaluation_sexual.json")
    print("✅ Evaluation completed!\n")


if __name__ == "__main__":
    main()

